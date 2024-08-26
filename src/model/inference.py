from model_loading import load_model

pipeline, tokenizer, text_encoder = load_model()
import torch
def generate_image(prompt):
    generator = torch.manual_seed(42)
    with torch.autocast("cuda"):
        image = pipeline(prompt, guidance_scale=7.5, num_inference_steps=100, generator=generator)["sample"][0]
    return image
