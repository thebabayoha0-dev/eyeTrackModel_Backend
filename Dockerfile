FROM python:3.11-slim

WORKDIR /app

# Prevent python buffer/cache issues
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app ./app
COPY models ./models
COPY main.py ./
COPY README.md ./
COPY .env.example ./

# Create runtime data directory
RUN mkdir -p /app/data/runs

ENV HOST=0.0.0.0
ENV PORT=8004

EXPOSE 8004

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]