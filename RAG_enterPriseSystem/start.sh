#!/bin/bash
# IP-SAKTI Sahayak Startup Script for Single-Container Deployment (e.g., Render)

# 1. Start the FastAPI backend in the background
# The Streamlit UI connects to this via http://localhost:8000 internally
echo "Starting FastAPI backend on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait a moment for the backend to start up
sleep 3

# 2. Start the Streamlit UI on the port provided by the platform (Render uses $PORT)
# Fallback to 8501 if $PORT is not set
STREAMLIT_PORT=${PORT:-8501}
echo "Starting Streamlit UI on port ${STREAMLIT_PORT}..."
streamlit run ui/app.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0
