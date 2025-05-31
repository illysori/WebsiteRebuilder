from PIL import Image, ImageDraw, ImageFont
import os

# Get the absolute path to the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define directories relative to this script
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
OUTPUT_DIR = os.path.join(BASE_DIR, "font_dataset")

# Text samples to render
TEXT_SAMPLES = [
    "The quick brown fox jumps over the lazy dog",
    "Sample Text",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "Iliad1"
]

# Image settings
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 60
FONT_SIZE = 48

def render_font_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for font_file in os.listdir(FONTS_DIR):
        if font_file.lower().endswith(('.ttf', '.otf')):
            font_name = os.path.splitext(font_file)[0]
            print(f"[INFO] Rendering samples for: {font_name}")

            font_path = os.path.join(FONTS_DIR, font_file)
            font_output_dir = os.path.join(OUTPUT_DIR, font_name)
            os.makedirs(font_output_dir, exist_ok=True)

            try:
                font = ImageFont.truetype(font_path, FONT_SIZE)
            except Exception as e:
                print(f"[ERROR] Failed to load font {font_file}: {e}")
                continue

            for i, text in enumerate(TEXT_SAMPLES):
                img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)  # white background
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = (IMAGE_WIDTH - text_w) // 2
                text_y = (IMAGE_HEIGHT - text_h) // 2
                draw.text((text_x, text_y), text, font=font, fill=0)  # black text

                img_path = os.path.join(font_output_dir, f"{font_name}_{i}.png")
                img.save(img_path)

    print("[INFO] Rendering complete.")

if __name__ == "__main__":
    render_font_images()
