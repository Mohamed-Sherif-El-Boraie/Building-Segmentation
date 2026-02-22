import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
import random

# --- Configuration ---
JSON_PATH = 'data/annotated_data/train/_annotations.coco.json'
IMAGES_PATH = 'data/annotated_data/train'
OUTPUT_PATH = 'data/image_tiles_plots'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def main():
    print(f"Loading annotations from {JSON_PATH}...")
    coco = COCO(JSON_PATH)
    
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} images.")

    # Get categories and generate random colors
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    # Map category ID to a random color (B, G, R)
    colors = {cat['id']: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cat in cats}

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        img_full_path = os.path.join(IMAGES_PATH, file_name)
        image = cv2.imread(img_full_path)

        if image is None:
            # print(f"Skipping missing image: {file_name}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Create a transparent overlay for the fills
        overlay = image.copy()

        for ann in anns:
            cat_id = ann['category_id']
            color = colors.get(cat_id, (255, 200, 200))
            
            # --- FIX: Universal Handling (RLE & Polygons) ---
            # 1. Convert annotation to binary mask
            mask = coco.annToMask(ann)
            
            # 2. Find contours (points) from the mask
            # RETR_EXTERNAL gets outer boundary, CHAIN_APPROX_SIMPLE reduces point count
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 3. Draw the outline (The "Segmentation Points")
            cv2.drawContours(image, contours, -1, color, 2)
            
            # 4. (Optional) Draw the semi-transparent fill on the overlay
            cv2.drawContours(overlay, contours, -1, color, -1)

        # Merge the overlay with the original image (Transparency effect)
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Save output
        save_path = os.path.join(OUTPUT_PATH, file_name)
        cv2.imwrite(save_path, image)
        
    print(f"Done! Segmentation plots saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()