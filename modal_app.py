import modal
import io
import base64
import os

app = modal.App("telegram-bot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "torch==2.3.1",
        "torchvision==0.18.1",
        "diffusers==0.31.0",
        "transformers==4.44.0",
        "accelerate==0.34.0",
        "safetensors",
        "sentencepiece",
        "Pillow",
        "numpy",
        "huggingface_hub>=0.25.0",
        "fastapi[standard]",
    ])
)

vol = modal.Volume.from_name("bot-models", create_if_missing=True)

@app.cls(
    gpu="A100",
    image=image,
    volumes={"/models": vol},
    timeout=600,
    secrets=[modal.Secret.from_name("hf-token")],
)
class Bot:

    @modal.enter()
    def setup(self):
        import torch
        from diffusers import FluxPipeline
        print("Loading model...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir="/models/flux-schnell",
            token=os.environ.get("HF_TOKEN"),
        ).to("cuda")
        print("Ready!")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, body: dict):
        prompt = body.get("prompt", "")
        width = body.get("width", 1024)
        height = body.get("height", 1024)
        if not prompt:
            return {"error": "No prompt"}
        try:
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=4,
                guidance_scale=0.0,
            ).images[0]
            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=95)
            return {"image": base64.b64encode(buf.getvalue()).decode()}
        except Exception as e:
            return {"error": str(e)}
