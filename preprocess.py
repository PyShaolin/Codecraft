import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (640, 256))  # Resize for consistency
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    
    return Image.fromarray(image)

if __name__ == "__main__":
    img_path = "test_image.png"
    processed_image = preprocess_image(img_path)
    processed_image.save("processed_test_image.png")
