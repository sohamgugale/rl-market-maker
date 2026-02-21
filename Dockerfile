FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models data
EXPOSE 8000 8001
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
