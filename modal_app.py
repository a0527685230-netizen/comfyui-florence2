
import modal
import io
import base64
import os
import traceback
import re

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
        "opencv-python-headless",
        "huggingface_hub>=0.25.0",
        "fastapi[standard]",
    ])
)

vol = modal.Volume.from_name("bot-models", create_if_missing=True)

COLOR_MAP = {
    "red":"#DC1414","dark red":"#8B0000","crimson":"#DC143C",
    "blue":"#1E32C8","dark blue":"#00008B","navy":"#000050","light blue":"#ADD8E6",
    "green":"#228B22","dark green":"#006400","lime":"#32CD32",
    "yellow":"#DCDC00","gold":"#FFD700",
    "orange":"#E66400","coral":"#FF6347",
    "purple":"#800080","violet":"#EE82EE","pink":"#DC5A82",
    "black":"#0A0A0A","white":"#F0F0F0","gray":"#808080","grey":"#808080",
    "brown":"#643214","beige":"#F5F5DC","cream":"#FFFDD0",
    "silver":"#C0C0C0","cyan":"#00CED1","turquoise":"#40E0D0",
}

def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _detect_color(text):
    t = text.lower()
    for name, hex_val in sorted(COLOR_MAP.items(), key=lambda x: -len(x[0])):
        if name in t:
            return _hex_to_rgb(hex_val)
    return None


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
        width  = (int(body.get("width",  1024)) // 16) * 16
        height = (int(body.get("height", 1024)) // 16) * 16
        if not prompt:
            return {"error": "No prompt"}
        try:
            img = self.pipe(
                prompt=prompt, width=width, height=height,
                num_inference_steps=4, guidance_scale=0.0,
            ).images[0]
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
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

        hf = os.environ.get("HF_TOKEN")
        print("Loading Florence2...")
        self.f2_proc = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True,
            cache_dir="/models/florence2", token=hf)
        self.f2_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", dtype=torch.float16,
            trust_remote_code=True, cache_dir="/models/florence2",
            token=hf, attn_implementation="eager").to("cuda")

        print("Loading Flux Fill...")
        self.fill_pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16,
            cache_dir="/models/flux-fill", token=hf).to("cuda")

        vol.commit()
        print("Edit Ready!")

    # ─── Florence2 helpers ───────────────────────────────────────
    def _f2(self, task, text, image):
        import torch
        inp = self.f2_proc(text=task+text, images=image,
                           return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        with torch.no_grad():
            ids = self.f2_model.generate(
                input_ids=inp["input_ids"],
                pixel_values=inp["pixel_values"],
                max_new_tokens=1024, num_beams=3, use_cache=False)
        txt = self.f2_proc.batch_decode(ids, skip_special_tokens=False)[0]
        return self.f2_proc.post_process_generation(
            txt, task=task, image_size=(image.width, image.height))

    def _seg_mask(self, image, text):
        from PIL import Image as P, ImageDraw
        task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        r = self._f2(task, text, image)
        mask = P.new("L", image.size, 0); draw = ImageDraw.Draw(mask)
        found = False
        if r and task in r:
            for pg in r[task].get("polygons", []):
                for poly in pg:
                    if len(poly) >= 6:
                        pts = [(poly[j], poly[j+1]) for j in range(0, len(poly), 2)]
                        draw.polygon(pts, fill=255); found = True
        return mask if found else None

    def _bbox_mask(self, image, text):
        from PIL import Image as P, ImageDraw
        task = "<OPEN_VOCABULARY_DETECTION>"
        r = self._f2(task, text, image)
        mask = P.new("L", image.size, 0); draw = ImageDraw.Draw(mask)
        found = False
        if r and task in r:
            for bb in r[task].get("bboxes", []):
                x1,y1,x2,y2 = [int(v) for v in bb]
                draw.rectangle([x1,y1,x2,y2], fill=255); found = True
        return mask if found else None

    def _get_mask(self, image, target, action):
        from PIL import ImageFilter
        import numpy as np
        mask = self._seg_mask(image, target)
        if mask is None:
            mask = self._bbox_mask(image, target)
        if mask is None:
            words = target.split()
            if len(words) > 1:
                mask = self._seg_mask(image, words[0]) or self._bbox_mask(image, words[0])
        if mask is None:
            return None
        mask = mask.filter(ImageFilter.MaxFilter(13))  # dilate
        mask = mask.filter(ImageFilter.GaussianBlur(3))
        if np.array(mask).max() < 10:
            return None
        return mask

    def _get_face_mask(self, image):
        from PIL import ImageFilter
        for query in ["face", "human face", "person face"]:
            m = self._bbox_mask(image, query)
            if m:
                return m.filter(ImageFilter.MaxFilter(31))
        return None

    # ─── Color change (image processing — no Flux Fill) ──────────
    def _apply_color(self, image, mask, color_rgb):
        import numpy as np
        from PIL import Image as P
        img = np.array(image).astype(np.float32) / 255.0
        msk = np.array(mask).astype(np.float32) / 255.0
        m3  = np.stack([msk]*3, axis=2)
        cr, cg, cb = [c/255.0 for c in color_rgb]
        lum = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        mx  = max(cr, cg, cb) or 1.0
        colored = np.stack([lum*(cr/mx), lum*(cg/mx), lum*(cb/mx)], axis=2)
        colored = np.clip(colored * mx / (np.max(colored)+1e-6) * mx, 0, 1)
        # blend
        result  = img*(1-m3) + colored*m3
        return P.fromarray((np.clip(result,0,1)*255).astype(np.uint8))

    # ─── Compositing ─────────────────────────────────────────────
    def _composite(self, original, generated, mask):
        import numpy as np
        from PIL import Image as P
        o = np.array(original).astype(np.float32)
        g = np.array(generated).astype(np.float32)
        m = np.array(mask).astype(np.float32)/255.0
        m3= np.stack([m]*3, axis=2)
        return P.fromarray((g*m3+o*(1-m3)).astype(np.uint8))

    def _restore_face(self, original, result, face_mask):
        """פנים מקוריות חוזרות בכל מקרה"""
        from PIL import Image as P, ImageFilter
        import numpy as np
        # blur שוליים של הפנים לחיבור חלק
        soft = face_mask.filter(ImageFilter.GaussianBlur(5))
        return self._composite(result, original, soft)

    # ─── Main endpoint ────────────────────────────────────────────
    @modal.fastapi_endpoint(method="POST")
    def edit(self, body: dict):
        import numpy as np
        from PIL import Image as P

        img_b64     = body.get("image", "")
        mask_target = body.get("mask_target", "")
        fill_prompt = body.get("fill_prompt", "")
        action      = body.get("action", "change")
        color_name  = body.get("color_name", "")
        orig_w      = int(body.get("orig_width",  1024))
        orig_h      = int(body.get("orig_height", 1024))

        if not img_b64 or not mask_target:
            return {"error": "Missing parameters"}

        try:
            pil = P.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB").resize((1024,1024))
            print(f"action={action} target={mask_target} fill={fill_prompt} color={color_name}")

            # 1. קבל מסכה
            mask = self._get_mask(pil, mask_target, action)
            if mask is None:
                return {"error": "לא זיהיתי את האובייקט: "+mask_target+". נסה לנסח אחרת."}

            # 2. זהה פנים מראש (כדי להחזיר אחר כך)
            face_mask = self._get_face_mask(pil)

            # 3. בצע עריכה
            if action == "color" and color_name:
                rgb = _detect_color(color_name)
                if rgb:
                    final = self._apply_color(pil, mask, rgb)
                else:
                    # fallback ל-Flux Fill אם הצבע לא מזוהה
                    final = self._run_flux_fill(pil, mask, fill_prompt or color_name+" shirt, photorealistic")
            else:
                final = self._run_flux_fill(pil, mask, fill_prompt)

            # 4. החזר פנים מקוריות
            if face_mask and action != "face":
                final = self._restore_face(pil, final, face_mask)

            # 5. החזר לגודל מקורי
            if orig_w > 0 and orig_h > 0 and (orig_w != 1024 or orig_h != 1024):
                final = final.resize((orig_w, orig_h), P.LANCZOS)

            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            return {"image": base64.b64encode(buf.getvalue()).decode()}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR:\n{tb}")
            return {"error": str(e), "traceback": tb[-500:]}

    def _run_flux_fill(self, image, mask, prompt):
        import numpy as np
        from PIL import Image as P
        steps   = 70 if "bare" in prompt or "no shirt" in prompt or "remove" in prompt else 50
        guidance= 55 if "bare" in prompt or "no shirt" in prompt or "remove" in prompt else 35
        gen = self.fill_pipe(
            prompt=prompt, image=P.fromarray(np.array(image)),
            mask_image=P.fromarray(np.array(mask), mode="L"),
            height=1024, width=1024,
            guidance_scale=guidance, num_inference_steps=steps,
        ).images[0]
        return self._composite(image, gen, mask)
