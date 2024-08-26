import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
import subprocess
import time

# Define the path to the FastAPI script (app.py)
app_path = "/home/ubuntu/text-to-img/src/Version2/app.py"  # Update this with the correct path

# Start the FastAPI server as a subprocess
fastapi_process = subprocess.Popen(["uvicorn", "src.Version2.app:app", "--host", "127.0.0.1", "--port", "8000"], cwd="/home/ubuntu/text-to-img")

# Wait for the server to start
time.sleep(2)

def generate_images_via_api(prompt, num_images, guidance_scale):
    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "num_images": int(num_images),
        "guidance_scale": float(guidance_scale)
    }

    # Make the request to the FastAPI endpoint
    api_url = "http://localhost:8000/generate"
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        images_data = response.json()["images"]
        images = [Image.open(BytesIO(base64.b64decode(img))) for img in images_data]
        return images
    else:
        return ["Error: " + response.json().get("detail", "Unknown error")]

# Define the Gradio interface
gr_interface = gr.Interface(
    fn=generate_images_via_api,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(1, 8, step=1, label="Number of Images", value=4),
        gr.Slider(1.0, 15.0, step=0.1, label="Guidance Scale", value=7.5)
    ],
    outputs=[gr.Gallery(label="Generated Images")],
    title="Stable Diffusion Image Generator",
    description="Generate images based on text prompts using a locally hosted FastAPI."
)

# Launch the Gradio interface
if __name__ == "__main__":
    try:
        gr_interface.launch()
    finally:
        # Ensure the FastAPI server is terminated when Gradio stops
        fastapi_process.terminate()
