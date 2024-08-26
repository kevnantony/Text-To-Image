import gradio as gr
# from src.model.inference import generate_image

from inference import generate_image

def gradio_interface(prompt):
    image = generate_image(prompt)
    return image

iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="image",
    title="Text-to-Image Generation",
    description="Generate images from text prompts using Stable Diffusion"
)

if __name__ == "__main__":
    iface.launch()
