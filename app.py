from flask import Flask, request, jsonify, render_template

from diffusers import StableDiffusionPipeline
import torch

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
        # Save the image and return its URL
        image_path = f"static/generated_images/{prompt.replace(' ', '_')}.png"  # Create a path based on prompt
        image.save(image_path)  # Save the image using PIL

        return jsonify(image_url=image_path)  # Return the URL of the generated image

    return jsonify(error="No prompt provided"), 400

def generate_image_from_prompt(prompt):
    # Generate an image based on the prompt
    images = pipeline(prompt).images
    return images[0]

if __name__ == "__main__":
    app.run(debug=False)