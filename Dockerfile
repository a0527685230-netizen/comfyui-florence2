FROM runpod/worker-comfyui:latest

RUN pip install timm einops "transformers>=4.41.0" --quiet

RUN git clone --depth=1 https://github.com/kijai/ComfyUI-Florence2.git /comfyui/custom_nodes/ComfyUI-Florence2

RUN pip install -r /comfyui/custom_nodes/ComfyUI-Florence2/requirements.txt --quiet || true
