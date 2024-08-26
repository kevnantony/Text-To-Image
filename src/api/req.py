import requests
import base64
from PIL import Image
from io import BytesIO

# Define the API endpoints
BASE_URL = "http://0.0.0.0:8000"
GENERATE_ENDPOINT = f"{BASE_URL}/generate"

# Test the GET endpoint
response = requests.get(BASE_URL)
print(response.json())

# Test the POST endpoint with a prompt as a query parameter
prompt = "A futuristic cityscape at sunset"
params = {"prompt": prompt, "num_inference_steps": 50, "guidance_scale": 7.5}
response = requests.post(GENERATE_ENDPOINT, params=params)

if response.status_code == 200:
    response_json = response.json()
    image_data = response_json.get("image")
    image_path = response_json.get("saved_image_path")

    if image_data:
        # Decode the base64 image data
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image.show()  # This will open the generated image

        # Save the image locally
        if image_path:
            image.save(f"local_{prompt.replace(' ', '_')}.png")
            print(f"Image saved locally as local_{prompt.replace(' ', '_')}.png")
    else:
        print("No image data found in response.")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
