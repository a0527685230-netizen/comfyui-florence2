FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /comfyui
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install -r requirements.txt --quiet

RUN git clone --depth=1 https://github.com/kijai/ComfyUI-Florence2.git \
    /comfyui/custom_nodes/ComfyUI-Florence2 && \
    pip install timm einops "transformers>=4.41.0" supervision --quiet && \
    (pip install -r /comfyui/custom_nodes/ComfyUI-Florence2/requirements.txt --quiet || true)

RUN pip install runpod requests --quiet

COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

RUN mkdir -p /comfyui/models/checkpoints /comfyui/models/LLM \
    /comfyui/models/sams /comfyui/output

CMD ["/start.sh"]
