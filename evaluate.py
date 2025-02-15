import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = TrOCRProcessor.from_pretrained("./models/fine_tuned_trocr")
model = VisionEncoderDecoderModel.from_pretrained("./models/fine_tuned_trocr").to(device)

dataset = load_dataset("IAM", split="test")
def preprocess_batch(batch):
    images = [processor(Image.open(img).convert("RGB"), return_tensors="pt").pixel_values for img in batch["image"]]
    labels = batch["text"]
    return images, labels

def evaluate_model(dataset, num_samples=100):
    total_samples = min(len(dataset), num_samples)
    correct = 0
    total = 0

    for i in tqdm(range(total_samples)):
        image_path = dataset[i]["image"]
        true_text = dataset[i]["text"]

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"\nGround Truth: {true_text}\nPredicted: {predicted_text}\n")

        if true_text.strip().lower() == predicted_text.strip().lower():
            correct += 1
        total += 1

    accuracy = correct / total * 100
    print(f"\nModel Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model(dataset)
