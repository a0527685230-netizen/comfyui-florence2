#!/bin/bash
echo "Starting ComfyUI..."
cd /comfyui
python main.py --listen 127.0.0.1 --port 8188 &
echo "Waiting for ComfyUI..."
until curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; do
    sleep 1
done
echo "ComfyUI ready!"
python -u /handler.py
