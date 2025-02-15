import os
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from flask_cors import CORS

device = "cuda" if torch.cuda.is_available() else "cpu"


model_path = "./models/fine_tuned_trocr"
if os.path.exists(model_path):
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
else:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def recognize_text(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return predicted_text
    except Exception as e:
        return str(e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
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
    
    return render_template("predict.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
