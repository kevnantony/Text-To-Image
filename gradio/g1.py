import gradio as gr
import requests
from io import BytesIO  # Ensure this is properly imported
from PIL import Image
from zipfile import ZipFile  # Ensure this is properly imported as well

# URL of the FastAPI server
API_URL = "http://localhost:8000/generate"

def generate_images(prompt, num_images, guidance_scale):
    # Prepare the payload
    payload = {
        "prompt": prompt,
        "num_images": num_images,
        "guidance_scale": guidance_scale
    }
    
    # Send POST request to FastAPI server
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        # If only one image, show it directly
        if num_images == 1:
            image_data = BytesIO(response.content)
            img = Image.open(image_data)
            return [img]
        else:
            # If multiple images, we expect a ZIP file
            img_list = []
            zip_data = BytesIO(response.content)
            with ZipFile(zip_data) as z:
                for file_name in z.namelist():
                    with z.open(file_name) as img_file:
                        img = Image.open(img_file)
                        img_list.append(img)
            return img_list
    else:
        return "Error: " + response.text

# Create Gradio interface
interface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(1, 4, step=1, label="Number of Images"),
        gr.Slider(1.0, 10.0, step=0.5, label="Guidance Scale")
    ],
    outputs=gr.Gallery(label="Generated Images")
)


# Launch the Gradio app
interface.launch(share=True)

