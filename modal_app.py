import modal, io, base64, os, traceback

app = modal.App("telegram-bot")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install([
        "torch>=2.4.0", "torchvision",
        "diffusers>=0.32.0",
        "transformers>=4.47.0,<5.0.0",
        "accelerate>=0.34.0", "safetensors",
        "sentencepiece", "timm", "einops",
        "Pillow", "numpy",
        "huggingface_hub>=0.25.0",
        "fastapi[standard]",
    ])
)

vol = modal.Volume.from_name("bot-models", create_if_missing=True)

COLOR_MAP = {
    "red":(220,30,30),"dark red":(139,0,0),"crimson":(220,20,60),
    "blue":(30,100,220),"dark blue":(0,0,180),"navy":(0,0,80),
    "light blue":(135,206,235),"green":(34,139,34),"dark green":(0,100,0),
    "lime":(50,205,50),"yellow":(220,220,0),"gold":(255,215,0),
    "orange":(230,120,0),"coral":(255,100,80),
    "purple":(128,0,128),"violet":(238,130,238),"pink":(220,90,130),
    "black":(15,15,15),"white":(240,240,240),
    "gray":(128,128,128),"grey":(128,128,128),
    "brown":(100,50,20),"beige":(245,245,220),
    "silver":(192,192,192),"cyan":(0,206,209),"turquoise":(64,224,208),
}

def _detect_color(text):
    t = text.lower()
    for name, rgb in sorted(COLOR_MAP.items(), key=lambda x: -len(x[0])):
        if name in t: return rgb
    return None


# ═══ יצירת תמונה מטקסט (Flux Schnell — לא משתנה) ══════════════

@app.cls(gpu="A100", image=image, volumes={"/models": vol},
         timeout=600, secrets=[modal.Secret.from_name("hf-token")])
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
        print("Bot Ready!")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, body: dict):
        prompt = body.get("prompt", "")
        width  = (int(body.get("width",  1024)) // 16) * 16
        height = (int(body.get("height", 1024)) // 16) * 16
        if not prompt: return {"error": "No prompt"}
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


# ═══ עריכת תמונה (SDXL Inpaint — חדש, טוב יותר להסרת בגדים) ════

@app.cls(gpu="A100", image=image, volumes={"/models": vol},
         timeout=600, secrets=[modal.Secret.from_name("hf-token")])
class ImageEdit:

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        from diffusers import StableDiffusionXLInpaintPipeline

        hf = os.environ.get("HF_TOKEN")

        print("Loading Florence2...")
        self.f2_proc = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True,
            cache_dir="/models/florence2", token=hf)
        self.f2_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base", dtype=torch.float16,
            trust_remote_code=True, cache_dir="/models/florence2",
            token=hf, attn_implementation="eager").to("cuda")

        print("Loading SDXL Inpaint...")
        self.inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            cache_dir="/models/sdxl-inpaint",
            use_safetensors=True,
        ).to("cuda")

        vol.commit()
        print("Edit Ready!")

    # ─── Florence2 ────────────────────────────────────────────────

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
        mask = P.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        found = False
        if r and task in r:
            for pg in r[task].get("polygons", []):
                for poly in pg:
                    if len(poly) >= 6:
                        pts = [(poly[j], poly[j+1]) for j in range(0, len(poly), 2)]
                        draw.polygon(pts, fill=255)
                        found = True
        return mask if found else None

    def _bbox_mask(self, image, text):
        from PIL import Image as P, ImageDraw
        task = "<OPEN_VOCABULARY_DETECTION>"
        r = self._f2(task, text, image)
        mask = P.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        found = False
        if r and task in r:
            for bb in r[task].get("bboxes", []):
                x1,y1,x2,y2 = [int(v) for v in bb]
                draw.rectangle([x1,y1,x2,y2], fill=255)
                found = True
        return mask if found else None

    def _get_mask(self, image, target):
        from PIL import ImageFilter
        import numpy as np
        mask = (self._seg_mask(image, target)
                or self._bbox_mask(image, target))
        if mask is None:
            w = target.split()
            if len(w) > 1:
                mask = (self._seg_mask(image, w[0])
                        or self._bbox_mask(image, w[0]))
        if mask is None: return None
        # דילציה קטנה בלבד — לא להחריג ידיים/פנים
        mask = mask.filter(ImageFilter.MaxFilter(9))
        mask = mask.filter(ImageFilter.GaussianBlur(3))
        return mask if np.array(mask).max() > 10 else None

    def _get_face_mask(self, image):
        from PIL import ImageFilter
        for q in ["face", "human face"]:
            m = self._bbox_mask(image, q)
            if m: return m.filter(ImageFilter.MaxFilter(25))
        return None

    # ─── עיבוד תמונה ─────────────────────────────────────────────

    def _composite(self, original, generated, mask):
        import numpy as np
        from PIL import Image as P
        o = np.array(original).astype(np.float32)
        g = np.array(generated).astype(np.float32)
        m = np.array(mask).astype(np.float32) / 255.0
        m3 = np.stack([m]*3, axis=2)
        return P.fromarray((g*m3 + o*(1-m3)).astype(np.uint8))

    def _restore_face(self, original, result, face_mask):
        from PIL import ImageFilter
        soft = face_mask.filter(ImageFilter.GaussianBlur(5))
        return self._composite(result, original, soft)

    def _apply_color(self, image, mask, rgb):
        import numpy as np
        from PIL import Image as P
        img = np.array(image).astype(np.float32) / 255.0
        m   = np.array(mask).astype(np.float32) / 255.0
        m3  = np.stack([m]*3, axis=2)
        cr, cg, cb = [c/255.0 for c in rgb]
        lum = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
        mx  = max(cr,cg,cb) or 1.0
        colored = np.clip(
            np.stack([lum*(cr/mx), lum*(cg/mx), lum*(cb/mx)], axis=2) * mx,
            0, 1)
        return P.fromarray((np.clip(img*(1-m3)+colored*m3,0,1)*255).astype(np.uint8))

    def _build_neg_prompt(self, action, fill_prompt):
        base = "(worst quality, low quality:1.4), (bad anatomy:1.3), blurry, artifacts, watermark, text"
        if action == "remove" or any(k in fill_prompt.lower()
                for k in ["bare","chest","skin","nude","shirtless","topless","naked"]):
            return base + ", (shirt:1.5), (t-shirt:1.5), (clothing:1.5), fabric, covered, dressed"
        return base

    # ─── SDXL Inpaint — הלב של המערכת ───────────────────────────

    def _sdxl_edit(self, image, mask, prompt, action):
        neg = self._build_neg_prompt(action, prompt)
        print(f"SDXL prompt: {prompt}")
        print(f"SDXL neg: {neg}")
        result = self.inpaint(
            prompt=prompt,
            negative_prompt=neg,
            image=image,
            mask_image=mask,
            height=1024,
            width=1024,
            strength=0.99,
            guidance_scale=9.0,
            num_inference_steps=50,
        ).images[0]
        return result

    # ─── endpoint ─────────────────────────────────────────────────

    @modal.fastapi_endpoint(method="POST")
    def edit(self, body: dict):
        from PIL import Image as P

        img_b64     = body.get("image", "")
        mask_target = body.get("mask_target", "")
        fill_prompt = body.get("fill_prompt", "")
        action      = body.get("action", "change")
        color_name  = body.get("color_name", "")
        orig_w      = int(body.get("orig_width", 1024))
        orig_h      = int(body.get("orig_height", 1024))

        if not img_b64 or not mask_target:
            return {"error": "Missing parameters"}

        try:
            pil = P.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB").resize((1024,1024))
            print(f"action={action} | target={mask_target} | color={color_name}")

            # 1. מסכה
            mask = self._get_mask(pil, mask_target)
            if mask is None:
                return {"error": f"לא זיהיתי: '{mask_target}'. נסח אחרת."}

            # 2. פנים (לשחזור אחר כך)
            face_mask = self._get_face_mask(pil)

            # 3. עריכה
            if action == "color" and color_name:
                rgb = _detect_color(color_name)
                if rgb:
                    final = self._apply_color(pil, mask, rgb)
                else:
                    final = self._sdxl_edit(pil, mask, fill_prompt, action)
            else:
                # SDXL Inpaint — רואה הקשר, מתאים עור, עוקב אחרי פרומפט
                final = self._sdxl_edit(pil, mask, fill_prompt, action)

            # 4. פנים מקוריות — תמיד
            if face_mask:
                final = self._restore_face(pil, final, face_mask)

            # 5. גודל מקורי
            if orig_w > 0 and orig_h > 0 and (orig_w != 1024 or orig_h != 1024):
                final = final.resize((orig_w, orig_h), P.LANCZOS)

            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=95)
            return {"image": base64.b64encode(buf.getvalue()).decode()}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"ERROR:\n{tb}")
            return {"error": str(e), "traceback": tb[-500:]}
