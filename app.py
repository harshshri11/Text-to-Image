from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Load the model
model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()  # Get JSON data
    prompt = data.get('prompt')  # Get the prompt from the JSON
    if prompt:
        image = generate_image_from_prompt(prompt)
        if image:
            # Convert the image to base64 without saving it
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Return the base64 encoded image data in a JSON response
            return jsonify(image_url=f"data:image/png;base64,{image_base64}"), 200
        else:
            return jsonify(error="Image generation failed"), 500  # Return error if generation failed
    return jsonify(error="No prompt provided"), 400  # Return error if no prompt

def generate_image_from_prompt(prompt):
    try:
        # Generate an image based on the prompt
        images = pipeline(prompt).images
        return images[0]  # Return the first image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use port 5000 as fallback
    app.run(host='0.0.0.0', port=port)
