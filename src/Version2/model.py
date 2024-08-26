import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torch import autocast

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, use_safetensors=False  # Change to False
)
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def generate_image(prompt, num_images=4, guidance_scale=7.5):
    images = []
    for _ in range(num_images):
        with autocast(device_type=device):
            result = pipe(prompt, guidance_scale=guidance_scale)
            images.append(result["images"][0])
    return images


