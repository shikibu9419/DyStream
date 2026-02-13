#!/bin/bash

# DyStream Streaming Server Startup Script

echo "Starting DyStream Streaming Server..."

# Set GPU device (change if needed)
export CUDA_VISIBLE_DEVICES=0

# Check if checkpoint exists
if [ ! -f "checkpoints/last.ckpt" ]; then
    echo "ERROR: Checkpoint not found at checkpoints/last.ckpt"
    echo "Please download the checkpoint first:"
    echo "  git clone https://huggingface.co/robinwitch/DyStream"
    echo "  cd DyStream && mv tools ../ && mv checkpoints ../ && cd .. && rm -rf DyStream"
    exit 1
fi

# Check if requirements are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "ERROR: FastAPI not installed"
    echo "Please install requirements:"
    echo "  pip install -r streaming_app/requirements_streaming.txt"
    exit 1
fi

# Start server
echo "Starting uvicorn server on http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

uvicorn streaming_app.main:app --host 0.0.0.0 --port 8000 --reload
