import modal
import io
import base64
import os
import traceback

app = modal.App("telegram-bot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "torch>=2.4.0",
        "torchvision",
        "diffusers>=0.32.0",
        "transformers>=4.47.0,<5.0.0",
        "accelerate>=0.34.0",
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
        width = int(body.get("width", 1024))
        height = int(body.get("height", 1024))
        # Flux requires multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
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
            dtype=torch.float16,
            trust_remote_code=True,
            cache_dir="/models/florence2",
            token=hf_token,
            attn_implementation="eager",
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

    def _run_florence2(self, task, text, image):
        import torch
        inputs = self.f2_processor(
            text=task + text,
            images=image,
            return_tensors="pt"
        ).to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            generated_ids = self.f2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                use_cache=False,
            )
        generated_text = self.f2_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return self.f2_processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

    def _segmentation_mask(self, image, text):
        from PIL import Image as PILImage, ImageDraw
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        result = self._run_florence2(task, text, image)
        mask = PILImage.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)
        found = False
        if result and task in result:
            for pg in result[task].get("polygons", []):
                for polygon in pg:
                    if len(polygon) >= 6:
                        pts = [(polygon[j], polygon[j+1]) for j in range(0, len(polygon), 2)]
                        draw.polygon(pts, fill=255)
                        found = True
        return mask if found else None

    def _bbox_mask(self, image, text):
        from PIL import Image as PILImage, ImageDraw
        task = "<OPEN_VOCABULARY_DETECTION>"
        result = self._run_florence2(task, text, image)
        mask = PILImage.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)
        found = False
        if result and task in result:
            for bbox in result[task].get("bboxes", []):
                x1, y1, x2, y2 = [int(v) for v in bbox]
                draw.rectangle([x1, y1, x2, y2], fill=255)
                found = True
        return mask if found else None

    def _dilate_mask(self, mask, pixels=10):
        from PIL import ImageFilter
        return mask.filter(ImageFilter.MaxFilter(size=pixels * 2 + 1))

    def _subtract_mask(self, base_mask, exclude_mask):
        import numpy as np
        from PIL import Image as PILImage
        b = np.array(base_mask).astype(np.float32)
        e = np.array(exclude_mask).astype(np.float32)
        result = np.clip(b - e * 2.0, 0, 255).astype(np.uint8)
        return PILImage.fromarray(result, mode="L")

    def _is_near_face_item(self, text):
        t = text.lower()
        return bool(__import__('re').search(
            r'shirt|t-shirt|tshirt|top|blouse|dress|jacket|coat|'
            r'sweater|hoodie|tank|בגד|חולצה|שמלה|מעיל|סוודר', t))

    def _get_mask(self, image, mask_target, action):
        import numpy as np
        from PIL import Image as PILImage, ImageFilter

        # נסה segmentation קודם, אחר כך bbox
        mask = self._segmentation_mask(image, mask_target)
        if mask is None:
            mask = self._bbox_mask(image, mask_target)
        if mask is None:
            # נסה מילה אחת בלבד אם הקלט ארוך
            words = mask_target.split()
            if len(words) > 1:
                mask = self._segmentation_mask(image, words[0])
                if mask is None:
                    mask = self._bbox_mask(image, words[0])
        if mask is None:
            return None

        # הרחב קצת את המסכה לכסות שפות
        mask = self._dilate_mask(mask, pixels=6)

        # הסר בגדים — החרג פנים מהמסכה
        if action == "remove" and self._is_near_face_item(mask_target):
            face_mask = self._bbox_mask(image, "face")
            if face_mask is None:
                face_mask = self._bbox_mask(image, "person face head")
            if face_mask:
                face_dilated = self._dilate_mask(face_mask, pixels=20)
                mask = self._subtract_mask(mask, face_dilated)

        # blur לשוליים חלקים
        mask = mask.filter(ImageFilter.GaussianBlur(radius=3))

        # וידוי שהמסכה לא ריקה אחרי ההחרגה
        arr = __import__('numpy').array(mask)
        if arr.max() < 10:
            return None

        return mask

    def _composite(self, original, generated, mask):
        import numpy as np
        from PIL import Image as PILImage
        o = np.array(original).astype(np.float32)
        g = np.array(generated).astype(np.float32)
        m = np.array(mask).astype(np.float32) / 255.0
        m3 = np.stack([m, m, m], axis=2)
        result = (g * m3 + o * (1 - m3)).astype(np.uint8)
        return PILImage.fromarray(result)

    @modal.fastapi_endpoint(method="POST")
    def edit(self, body: dict):
        import numpy as np
        from PIL import Image as PILImage

        image_b64 = body.get("image", "")
        mask_target = body.get("mask_target", "")
        fill_prompt = body.get("fill_prompt", "")
        action = body.get("action", "change")
        orig_width = int(body.get("orig_width", 1024))
        orig_height = int(body.get("orig_height", 1024))

        if not image_b64 or not mask_target or not fill_prompt:
            return {"error": "Missing parameters"}

        try:
            pil_image = PILImage.open(
                io.BytesIO(base64.b64decode(image_b64))
            ).convert("RGB").resize((1024, 1024))

            print(f"Action={action} Target={mask_target} Fill={fill_prompt}")

            mask_pil = self._get_mask(pil_image, mask_target, action)

            if mask_pil is None:
                return {"error": "לא הצלחתי לזהות: " + mask_target + ". נסה לנסח אחרת."}

            # פרמטרים לפי סוג פעולה
            if action == "remove":
                steps = 70
                guidance = 50
            elif action == "add":
                steps = 50
                guidance = 30
            else:
                steps = 50
                guidance = 35

            img_np = np.array(pil_image, dtype=np.uint8)
            mask_np = np.array(mask_pil, dtype=np.uint8)
            clean_image = PILImage.fromarray(img_np)
            clean_mask = PILImage.fromarray(mask_np, mode="L")

            generated = self.fill_pipe(
                prompt=fill_prompt,
                image=clean_image,
                mask_image=clean_mask,
                height=1024,
                width=1024,
                guidance_scale=guidance,
                num_inference_steps=steps,
            ).images[0]

            final = self._composite(pil_image, generated, mask_pil)

            # החזרה לגודל המקורי
            if orig_width > 0 and orig_height > 0:
                final = final.resize((orig_width, orig_height), PILImage.LANCZOS)

            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            return {"image": base64.b64encode(buf.getvalue()).decode()}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR:\n{tb}")
            return {"error": str(e), "traceback": tb[-500:]}
