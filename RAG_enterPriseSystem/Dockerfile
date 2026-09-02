# IP-SAKTI Sahayak — Prototype Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for PDF/office parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose both ports (Render will bind to $PORT for Streamlit)
EXPOSE 8000 8501

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
