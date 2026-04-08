FROM python:3.11-slim

LABEL maintainer="MLOps Credit Fraud Detection Team"
LABEL description="FastAPI serving for credit card fraud detection"

WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY configs/ configs/
COPY src/ src/
COPY codes/ codes/
COPY models/ models/
COPY model.joblib* ./

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
