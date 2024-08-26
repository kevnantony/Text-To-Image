from fastapi import HTTPException
from inference import generate_image

def generate_image_from_text(prompt: str):
    try:
        image = generate_image(prompt)
        # Convert image to a response-friendly format, e.g., base64 string
        image_base64 = convert_image_to_base64(image)
        return {"image": image_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def convert_image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
