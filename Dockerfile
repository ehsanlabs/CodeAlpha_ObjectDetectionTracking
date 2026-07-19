FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8n weights at build time (baked into image)
RUN wget -q https://github.com/ultralytics/assets/releases/latest/download/yolov8n.pt -O yolov8n.pt

COPY . .

EXPOSE 10000
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 300 --workers 1 web_app:app
