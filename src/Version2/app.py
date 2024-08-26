from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from torch import autocast
from PIL import Image
import io


app = FastAPI()

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, use_safetensors=False
)
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

class ImageGenerationRequest(BaseModel):
    prompt: str
    num_images: int = 1
    guidance_scale: float = 7.5

def generate_image(prompt: str, num_images: int = 1, guidance_scale: float = 7.5) -> List[Image.Image]:
    images = []
    for _ in range(num_images):
        with autocast(device_type=device):
            result = pipe(prompt, guidance_scale=guidance_scale)
            images.append(result["images"][0])
    return images

@app.post("/generate")
async def generate(request: ImageGenerationRequest):
    try:
        images = generate_image(request.prompt, request.num_images, request.guidance_scale)
        if len(images) == 1:
            image = images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")
        else:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w") as zip_file:
                for i, image in enumerate(images):
                    img_buf = io.BytesIO()
                    image.save(img_buf, format="PNG")
                    img_buf.seek(0)
                    zip_file.writestr(f"image_{i+1}.png", img_buf.getvalue())
            zip_buf.seek(0)
            return StreamingResponse(zip_buf, media_type="application/zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


