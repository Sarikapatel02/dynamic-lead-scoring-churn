FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

ENV DJANGO_SETTINGS_MODULE=lead_scoring_engine.settings
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "lead_scoring_engine/manage.py", "runserver", "0.0.0.0:8000"]
