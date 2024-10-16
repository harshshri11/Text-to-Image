from flask import Flask, request, jsonify, render_template, send_from_directory
from diffusers import StableDiffusionPipeline
import torch
import os

app = Flask(__name__)

# Load the model
model_id = "stabilityai/stable-diffusion-2-1"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Ensure the directory for generated images exists
generated_images_dir = "static/generated_images"
os.makedirs(generated_images_dir, exist_ok=True)  # Create the directory if it doesn't exist

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
            # Save the image and return its URL
            image_filename = f"{prompt.replace(' ', '_')}.png"  # Create a file name based on prompt
            image_path = os.path.join(generated_images_dir, image_filename)  # Full path for saving
            image.save(image_path)  # Save the image using PIL

            return jsonify(image_url=image_path), 200  # Return the URL of the generated image
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

# Serve static files manually if needed in production
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)
