import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from preprocess import preprocess_image

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

def recognize_text(image_path):
    image = preprocess_image(image_path)
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_text

if __name__ == "__main__":
    image_path = "test_image.png"
    result = recognize_text(image_path)
    print("Recognized Text:", result)
