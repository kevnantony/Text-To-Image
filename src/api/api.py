from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image
import logging
import os

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create a directory to save images if it doesn't exist
IMAGE_SAVE_DIR = "saved_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Initialize the Stable Diffusion Pipeline
try:
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe.to("cuda")
    logging.info("Pipeline initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing pipeline: {str(e)}")
    raise

class Prompt(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Text-to-Image API"}

@app.post("/generate")
async def generate_image(
    prompt: str, 
    num_inference_steps: int = Query(50, gt=0), 
    guidance_scale: float = Query(7.5, gt=0)
):
    try:
        logging.info(f"Received prompt: {prompt}")
        # Generate the image
        result = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        
        logging.info(f"Pipeline output: {result}")

        if not hasattr(result, 'images') or len(result.images) == 0:
            raise ValueError("No images found in pipeline output")

        image = result.images[0]
        
        # Save the image to a file
        image_filename = os.path.join(IMAGE_SAVE_DIR, f"{prompt.replace(' ', '_')}.png")
        image.save(image_filename)
        logging.info(f"Image saved to {image_filename}")
        
        # Convert the image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"image": img_str, "saved_image_path": image_filename}
    
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
