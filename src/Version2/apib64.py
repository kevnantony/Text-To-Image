from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from torch import autocast
from PIL import Image
import io
import base64

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
    num_images: int = 4
    guidance_scale: float = 7.5

def generate_image(prompt: str, num_images: int = 4, guidance_scale: float = 7.5) -> List[str]:
    images = []
    for _ in range(num_images):
        with autocast(device_type=device):
            result = pipe(prompt, guidance_scale=guidance_scale)
            image = result["images"][0]
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return images

@app.post("/generate")
async def generate(request: ImageGenerationRequest):
    try:
        images = generate_image(request.prompt, request.num_images, request.guidance_scale)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

