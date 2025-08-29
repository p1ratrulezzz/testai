import torch
from transformers import pipeline
from dotenv import load_dotenv
import os

# --- 1. Initialization ---
# Load environment variables from .env file
load_dotenv()
token = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f"Using device: {device}")

# Initialize the text generation pipeline
print("Loading model... This may take some time.")
pipe = pipeline(
    "text-generation",
    model="google/gemma-2b-it",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    token=token,
)
print("Model loaded successfully!")

# --- 2. Settings and Chat History ---
# Generation arguments
generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": True,
}

# List to store the conversation history, primed for English
messages = [
    {"role": "user", "content": "You are a helpful assistant. Always respond in Russian."},
    {"role": "model", "content": "Understood. I will always respond in Russian."}
]

# --- 3. Main Chat Loop ---
print("\n--- Interactive Agent ---")
print('Enter your question. Type "exit" or "quit" to leave.')

while True:
    # Get user input
    user_input = input("You: ")

    # Check if the user wants to exit
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    # Add user message to history
    messages.append({"role": "user", "content": user_input})

    # Get response from the model
    print("Agent: ...")
    output = pipe(messages, **generation_args)
    model_response = output[0]['generated_text'].strip()

    # Add model response to history
    messages.append({"role": "model", "content": model_response})

    # Print the model's response
    print(f"Agent: {model_response}")
