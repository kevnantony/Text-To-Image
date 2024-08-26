import gradio as gr
import sys
import os

# Add the path to the cdk directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cdk')))

from cdk  TextToImg import StableDiffusionModel

# Initialize the StableDiffusionModel with the desired model ID
model_id1 = "dreamlike-art/dreamlike-diffusion-1.0"
sd_model = StableDiffusionModel(model_id1)

# Define the function to be used by Gradio, wrapping the generate_image_interface method
def generate_image_gradio(prompt, negative_prompt, num_inference_steps=50, width=640):
    image1, image2 = sd_model.generate_image_interface(
        prompt, negative_prompt, num_inference_steps, width
    )
    return image1, image2

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_image_gradio,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Textbox(lines=2, placeholder="Enter negative prompt here...", label="Negative Prompt"),
        gr.Slider(minimum=10, maximum=150, step=1, value=50, label="Number of Inference Steps"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=640, label="Width"),
    ],
    outputs=[gr.Image(type="pil", label="Generated Image 1"), gr.Image(type="pil", label="Generated Image 2")],
    title="Stable Diffusion Image Generator",
    description="Generate images using the Stable Diffusion model with custom prompts."
)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()

