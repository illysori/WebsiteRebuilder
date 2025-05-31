import argparse
import numpy as np
import json
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

def preprocess_image(image_path):
    pil_img = Image.open(image_path).convert('L')
    pil_img = pil_img.resize((105, 105))
    img = img_to_array(pil_img)
    img = np.expand_dims(img, axis=0)
    img = img.astype("float") / 255.0
    return img

def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--model", required=True, help="Path to trained model")
    ap.add_argument("--label_map", required=True, help="Path to label map JSON file")
    args = vars(ap.parse_args())

    print("[INFO] Loading model...")
    model = load_model(args["model"])

    print("[INFO] Loading label map...")
    label_map = load_label_map(args["label_map"])
    label_map = {int(k): v for k, v in label_map.items()}  # Convert keys to int

    print("[INFO] Preprocessing image...")
    image = preprocess_image(args["image"])

    print("[INFO] Making prediction...")
    prediction = model.predict(image)
    label = int(np.argmax(prediction))
    print(f"Predicted font: {label_map.get(label, 'Unknown')}")

if __name__ == "__main__":
    main()
