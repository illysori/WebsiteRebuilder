import cv2
import pytesseract
import json
import os
from PIL import Image

# Optional: Set path to tesseract manually (only for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Step 1: Load the image ===
image_path = "input/screenshot.png"  # adjust if needed
img = cv2.imread(image_path)

# === Step 2: Convert to grayscale and preprocess ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# === Step 3: Run OCR with detailed output ===
data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

# === Step 4: Extract and store results ===
results = []
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    conf = int(data['conf'][i])
    
    # Filter out low-confidence and empty results
    if text and conf > 60:
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        # Draw green rectangle around detected text (debug)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add to results list
        results.append({
            "text": text,
            "bbox": [x, y, w, h],
            "confidence": conf
        })

# === Step 5: Save visual output and JSON ===
os.makedirs("output", exist_ok=True)

cv2.imwrite("output/ocr_result.png", img)  # Image with boxes
with open("output/ocr_data.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Processed {len(results)} text items.")
print("📦 Output saved to: output/ocr_result.png and output/ocr_data.json")
