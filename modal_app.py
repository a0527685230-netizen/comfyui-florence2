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
        "diffusers>=0.32.0",
        "transformers>=4.47.0,<5.0.0",
        "accelerate==0.34.0",
        "safetensors",
        "sentencepiece",
        "timm",
        "einops",
        "Pillow",
        "numpy",
        "huggingface_hub>=0.25.0",
        "fastapi[standard]",
    ])
)

vol = modal.Volume.from_name("bot-models", create_if_missing=True)

# ===== יצירת תמונה מטקסט =====
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
        print("Loading Flux Schnell...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            cache_dir="/models/flux-schnell",
            token=os.environ.get("HF_TOKEN"),
        ).to("cuda")
        vol.commit()
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


# ===== עריכת תמונה =====
@app.cls(
    gpu="A100",
    image=image,
    volumes={"/models": vol},
    timeout=600,
    secrets=[modal.Secret.from_name("hf-token")],
)
class ImageEdit:

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from diffusers import FluxFillPipeline

        hf_token = os.environ.get("HF_TOKEN")

        print("Loading Florence2...")
        self.f2_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            trust_remote_code=True,
            cache_dir="/models/florence2",
            token=hf_token,
        )
        self.f2_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir="/models/florence2",
            token=hf_token,
        ).to("cuda")

        print("Loading Flux Fill...")
        self.fill_pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
            cache_dir="/models/flux-fill",
            token=hf_token,
        ).to("cuda")

        vol.commit()
        print("Edit Ready!")

    def _get_mask(self, image, object_text):
        import torch
        from PIL import Image as PILImage, ImageDraw

        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        inputs = self.f2_processor(
            text=task + object_text,
            images=image,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = self.f2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.f2_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        result = self.f2_processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        mask = PILImage.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)

        if result and task in result:
            for polygon_group in result[task].get("polygons", []):
                for polygon in polygon_group:
                    if len(polygon) >= 6:
                        points = [(polygon[j], polygon[j+1]) for j in range(0, len(polygon), 2)]
                        draw.polygon(points, fill=255)

        return mask

    @modal.fastapi_endpoint(method="POST")
    def edit(self, body: dict):
        from PIL import Image as PILImage
        import torch

        image_b64 = body.get("image", "")
        object_text = body.get("object", "")
        edit_prompt = body.get("prompt", "")

        if not image_b64 or not object_text or not edit_prompt:
            return {"error": "Missing parameters"}

        try:
            image = PILImage.open(
                io.BytesIO(base64.b64decode(image_b64))
            ).convert("RGB").resize((1024, 1024))

            mask = self._get_mask(image, object_text)

            result = self.fill_pipe(
                prompt=edit_prompt,
                image=image,
                mask_image=mask,
                height=1024,
                width=1024,
                guidance_scale=30,
                num_inference_steps=50,
            ).images[0]

            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=95)
            return {"image": base64.b64encode(buf.getvalue()).decode()}

        except Exception as e:
            return {"error": str(e)}
