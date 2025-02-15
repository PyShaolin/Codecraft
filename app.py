import os
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from flask_cors import CORS

# Check if CUDA is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model path
model_path = "./models/fine_tuned_trocr"

try:
    if os.path.exists(model_path):
        processor = TrOCRProcessor.from_pretrained(model_path, use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    else:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)  # Stop execution if model fails to load

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for all routes

# Upload directory setup
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def recognize_text(image_path):
    """Process image and extract text using TrOCR"""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return predicted_text
    except Exception as e:
        return f"Error processing image: {e}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return recognized text"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate file extension
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid file format. Only PNG, JPG, and JPEG allowed."}), 400

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Recognize text
    predicted_text = recognize_text(file_path)
    
    return render_template("predict.html", text=predicted_text, image_url=f"/uploads/{file.filename}")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)

