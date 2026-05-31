import runpod, json, time, base64, requests, uuid

COMFYUI_URL = "http://127.0.0.1:8188"

def upload_image(image_data, filename):
    img_bytes = base64.b64decode(image_data)
    files = {"image": (filename, img_bytes, "image/png"), "overwrite": (None, "true")}
    requests.post(f"{COMFYUI_URL}/upload/image", files=files)

def queue_workflow(workflow):
    client_id = str(uuid.uuid4())
    data = json.dumps({"prompt": workflow, "client_id": client_id})
    r = requests.post(f"{COMFYUI_URL}/prompt", data=data)
    return r.json()["prompt_id"]

def wait_for_completion(prompt_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        history = r.json()
        if prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("completed", False):
                return True
            if status.get("status_str") == "error":
                return False
        time.sleep(1)
    return False

def get_images(prompt_id):
    r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    history = r.json()
    if prompt_id not in history:
        return []
    images = []
    for node_output in history[prompt_id]["outputs"].values():
        if "images" in node_output:
            for img in node_output["images"]:
                img_r = requests.get(f"{COMFYUI_URL}/view",
                    params={"filename": img["filename"],
                            "subfolder": img.get("subfolder", ""),
                            "type": img["type"]})
                images.append(base64.b64encode(img_r.content).decode())
    return images

def handler(job):
    job_input = job["input"]
    workflow = job_input.get("workflow")
    images_input = job_input.get("images", [])
    if not workflow:
        return {"error": "No workflow provided"}
    for img in images_input:
        upload_image(img["image"], img["name"])
    prompt_id = queue_workflow(workflow)
    if not wait_for_completion(prompt_id):
        return {"error": "Workflow failed or timed out"}
    output_images = get_images(prompt_id)
    if not output_images:
        return {"error": "No output images"}
    return {"images": [{"data": img} for img in output_images]}

runpod.serverless.start({"handler": handler})
