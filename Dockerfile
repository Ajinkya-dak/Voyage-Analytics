# Use a slim Python base image
FROM python:3.10-slim

# 1. Set workdir
WORKDIR /app

# 2. Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3. Copy application code and model
COPY . .

# 4. Expose port
EXPOSE 5000

# 5. Default command
CMD ["python", "app.py"]
