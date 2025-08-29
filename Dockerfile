# Dockerfile for Telegram Bot
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY telegram_bot.py requirements.txt ./

# Expose port if needed (not required for Telegram bot)
# EXPOSE 8000

# Set environment variables (can be overridden at runtime)
ENV HUGGING_FACE_HUB_TOKEN=""
ENV TELEGRAM_BOT_TOKEN=""
ENV ADMINS=""
ENV DATABASE_URL="postgresql://botuser:botpassword@postgres/telegram_bot"

# Run the bot
CMD ["python", "telegram_bot.py"]
