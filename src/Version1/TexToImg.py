#!/usr/bin/env python
# coding: utf-8

#diffusers is a hugging face page for using diffusion models from huggingface hub
# get_ipython().system('pip install diffusers transformers accelerate')


# In[2]:
# stable_diffusion_module.py

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

class StableDiffusionModel:
    def __init__(self, model_id, device="cuda"):
        self.model_id = model_id
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16, use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, prompt, **params):
        images = self.pipe(prompt, **params).images
        self._display_images(images)
        return images

    def _display_images(self, images):
        num_images = len(images)
        if num_images > 1:
            fig, ax = plt.subplots(nrows=1, ncols=num_images)
            for i in range(num_images):
                ax[i].imshow(images[i])
                ax[i].axis("off")
        else:
            plt.imshow(images[0])
            plt.axis("off")
        plt.tight_layout()

    def generate_image_interface(self, prompt, negative_prompt, num_inference_steps=50, width=640):
        params = {
            'prompt': prompt,
            'num_inference_steps': num_inference_steps,
            'num_images_per_prompt': 2,
            'height': int(1.2 * width),
            'width': width,
            'negative_prompt': negative_prompt
        }
        images = self.pipe(**params).images
        return images[0], images[1]

# Example usage:
if __name__ == "__main__":
    model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
    sd_model = StableDiffusionModel(model_id1)

    # Example prompts
    prompt1 = "dreamlikeart, a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin, extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans"
    prompt2 = "dreamlike, Goddess coming down from the heaven with a weapon in one hand and other hand in the pose of blessing. Anger and divine energy reflecting from her eyes. She is in the form of a soldier and savior coming to protect the world from misery. She is accompanied by her tiger. Make sure to keep it cinematic and color to be golden iris"

    # Generate images
    sd_model.generate_image(prompt1)
    sd_model.generate_image(prompt2)
