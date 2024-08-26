import gradio as gr
import requests
import base64
from PIL import Image
from io import BytesIO

# Define the API endpoints
BASE_URL = "http://0.0.0.0:8000"
GENERATE_ENDPOINT = f"{BASE_URL}/generate"

# Function to generate image from prompt and handle saving
def generate_image(prompt, num_inference_steps, guidance_scale):
    try:
        # Call the FastAPI endpoint
        response = requests.post(
            GENERATE_ENDPOINT, 
            params={
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
        )
        
        # Check if the response is successful
        if response.status_code == 200:
            response_json = response.json()
            image_data = response_json.get("image")
            image_path = response_json.get("saved_image_path")
            
            if image_data:
                # Decode the base64 image data
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                
                # Save the image locally
                if image_path:
                    local_image_path = f"local_{prompt.replace(' ', '_')}.png"
                    image.save(local_image_path)
                    print(f"Image saved locally as {local_image_path}")
                
                return image
            else:
                return "No image data found in response."
        else:
            return f"Error: {response.status_code}\n{response.json()}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Text Prompt", placeholder="Enter your prompt here"),
        gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Number of Inference Steps"),
        gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=7.5, label="Guidance Scale")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Text-to-Image Generator",
    description="Generate images from text prompts using a Stable Diffusion model."
)

interface.launch(share='True')
