# Use Python 3.12.3 base image
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run your bot
CMD ["python", "no_db_bot_02.py"]
