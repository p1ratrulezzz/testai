import os
import torch
from transformers import pipeline
from dotenv import load_dotenv
import logging
import re
import asyncio
import json
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from sqlalchemy import create_engine, Column, String, Text, Boolean, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
import requests

torch.set_float32_matmul_precision('high')

# --- 1. Initialization ---
# Load environment variables from .env file
load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

if not telegram_token:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print(f"Using device: {device}")

# Initialize the text generation pipeline
print("Loading model... This may take some time.")
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    token=token,
)


# --- 2. Settings and Chat History ---
# Generation arguments
generation_args = {
    "max_new_tokens": 4096,
    "return_full_text": False,
    "temperature": 0.3,
    "do_sample": True,
}

# Max history size
MAX_HISTORY_SIZE = 20

# Dictionary to store conversation history per chat_id
chat_histories = {}

# Set to track processing chats
chat_processing = set()

# Global lock for serializing generations
global_lock = asyncio.Lock()

# --- 3. Database Models ---
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_histories'
    id = Column(Integer, primary_key=True)
    chat_id = Column(String(50), unique=True)
    history = Column(Text)

class Permission(Base):
    __tablename__ = 'permissions'
    id = Column(Integer, primary_key=True)
    chat_id = Column(String(50), unique=True)
    allowed = Column(Boolean, default=False)

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://botuser:botpassword@localhost/telegram_bot')
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# Admins
admins = os.getenv('ADMINS', '').split(',')

# --- 4. Helper Functions ---
def get_chat_history(chat_id: int):
    session = Session()
    try:
        record = session.query(ChatHistory).filter_by(chat_id=str(chat_id)).first()
        if record:
            return json.loads(record.history)
        # Initialize with priming messages
        default_history = [
            {"role": "user", "content": "You are a cynical assistant who responds in a slightly rude tone. Always respond in Russian. Use Markdown formatting when appropriate."},
            {"role": "model", "content": "Understood. I will be cynical and slightly rude while responding in Russian with Markdown."}
        ]
        return default_history
    finally:
        session.close()

def save_chat_history(chat_id: int, history):
    session = Session()
    try:
        record = session.query(ChatHistory).filter_by(chat_id=str(chat_id)).first()
        if record:
            record.history = json.dumps(history)
        else:
            record = ChatHistory(chat_id=str(chat_id), history=json.dumps(history))
            session.add(record)
        session.commit()
    finally:
        session.close()

def check_permission(chat_id: int):
    session = Session()
    try:
        perm = session.query(Permission).filter_by(chat_id=str(chat_id)).first()
        return perm and perm.allowed
    finally:
        session.close()

def set_permission(chat_id: int, allowed=True):
    session = Session()
    try:
        perm = session.query(Permission).filter_by(chat_id=str(chat_id)).first()
        if perm:
            perm.allowed = allowed
        else:
            perm = Permission(chat_id=str(chat_id), allowed=allowed)
            session.add(perm)
        session.commit()
    finally:
        session.close()

async def generate_response(messages):
    # Get response from the model
    output = await asyncio.to_thread(lambda: pipe(messages, **generation_args))
    model_response = output[0]['generated_text'].strip()
    return model_response

def trim_history(messages, max_size=MAX_HISTORY_SIZE):
    """Trim conversation history to max_size, keeping priming messages."""
    if len(messages) > max_size:
        # Keep first 2 priming messages + last (max_size - 2) messages
        messages[:] = messages[:2] + messages[-(max_size - 2):]

# --- 5. Telegram Bot Handlers ---
async def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Привет! Я ваш помощник. Задайте вопрос!')

async def help_command(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text('Введите сообщение, и я отвечу на него.')

# /permit command removed - bot now responds only to mentions in groups

async def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    user_message = update.message.text
    chat_id = update.effective_chat.id

    print(f"[DEBUG] User message: {user_message}")

    # Check permission for private chats (only admins)
    chat = update.effective_chat
    user_id = update.effective_user.id
    if chat.type == 'private':
        if str(user_id) not in admins:
            return  # Ignore private messages from non-admins

    # Check for mentions or replies in group chats
    if chat.type != 'private':
        bot = await context.bot.get_me()
        bot_username = bot.username.lower()
        user_message_lower = user_message.lower()
        is_mentioned = '@' + bot_username in user_message_lower or any(
            entity.type == 'mention' and update.message.text[entity.offset:entity.offset + entity.length].lower() == '@' + bot_username
            for entity in (update.message.entities or [])
        )

        # Check if message is reply to bot
        reply_to = update.message.reply_to_message
        is_reply_to_bot = reply_to and reply_to.from_user and reply_to.from_user.id == context.bot.id

        if not is_mentioned and not is_reply_to_bot:
            return  # Ignore messages in groups without mention or reply

        # Check if message is reply to bot
        reply_to = update.message.reply_to_message
        is_reply_to_bot = reply_to and reply_to.from_user and reply_to.from_user.id == context.bot.id

        if not is_mentioned and not is_reply_to_bot:
            return  # Ignore messages in groups without mention or reply
            return  # Ignore messages in groups without permission or mention

    # Check if already processing this chat
    if chat_id in chat_processing:
        await update.message.reply_text("подожди я уже думаю")
        return

    # Mark as processing
    chat_processing.add(chat_id)
    try:
        # Get or initialize chat history
        messages = get_chat_history(chat_id)

        # Add user message to history
        messages.append({"role": "user", "content": user_message})

        # Acquire global lock for generation
        await global_lock.acquire()

        # Send typing action to show bot is thinking
        await update.effective_chat.send_chat_action(ChatAction.TYPING)

        print("Generating response...")

        # Generate response
        model_response = await generate_response(messages)

        # Add model response to history
        messages.append({"role": "model", "content": model_response})

        # Trim history to prevent memory overflow
        trim_history(messages)

        # Save history to database
        save_chat_history(chat_id, messages)

        # Send typing before response
        await update.effective_chat.send_chat_action(ChatAction.TYPING)

        # Send response back (split if too long)
        MAX_MESSAGE_LENGTH = 4000  # Telegram limit
        print(f"[DEBUG] Sending response, length: {len(model_response)}")
        if len(model_response) > MAX_MESSAGE_LENGTH:
            parts = [model_response[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(model_response), MAX_MESSAGE_LENGTH)]
            for i, part in enumerate(parts):
                try:
                    await update.message.reply_text(part, parse_mode='Markdown')
                    print(f"[DEBUG] Sent part {i+1}/{len(parts)}")
                except Exception as e:
                    print(f"[DEBUG] Failed to send part {i+1} with Markdown: {e}")
                    await update.message.reply_text(part)
        else:
            try:
                await update.message.reply_text(model_response, parse_mode='Markdown')
                print("[DEBUG] Response sent successfully")
            except Exception as e:
                print(f"[DEBUG] Failed to send with Markdown: {e}")
                await update.message.reply_text(model_response)
    finally:
        global_lock.release()
        chat_processing.remove(chat_id)

async def error_handler(update: Update, context: CallbackContext) -> None:
    """Log the error and send a telegram message to notify the developer."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(telegram_token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    # No /permit command - bot responds to mentions only

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # error handler
    application.add_error_handler(error_handler)

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
