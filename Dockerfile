FROM python:3.10-slim

WORKDIR /app

# OpenCV needs these system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n model weights at build time
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy app files
COPY . .

# Hugging Face Spaces requires port 7860
EXPOSE 7860
ENV PORT=7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "web_app:app"]
