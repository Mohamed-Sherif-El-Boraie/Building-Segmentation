import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from PIL import Image
import os
from pathlib import Path

# Configuration
images_path = "data/annotated_data/train"
output_image = "data/annotated_data/full_image.png"
img_size = 256
images_per_row = 14
total_images = 196

# Load all images
images = []
for filename in sorted(os.listdir(images_path)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(os.path.join(images_path, filename))
        images.append(img)

print(f"Loaded {len(images)} images")

# Create output image (14x14 grid)
rows = total_images // images_per_row
output_size = img_size * images_per_row
full_image = Image.new('RGB', (output_size, output_size))

# Paste images into grid
for idx, img in enumerate(images):
    row = idx // images_per_row
    col = idx % images_per_row
    x = col * img_size
    y = row * img_size
    full_image.paste(img, (x, y))

# Create output directory if needed
os.makedirs(os.path.dirname(output_image), exist_ok=True)

# Save
full_image.save(output_image)
print(f"Saved to {output_image}")
print(f"Output size: {full_image.size}")