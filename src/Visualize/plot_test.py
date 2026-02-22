import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


# import os
# import json
# import cv2
# import numpy as np
# from pycocotools.coco import COCO
# import random

# # --- Configuration ---
# # Changed extensions from .tif to .png
# FULL_IMAGE_PATH = 'data/annotated_data/full_image.png'
# JSON_PATH = 'data/output/1_morphological_coco.json'
# OUTPUT_PATH = 'data/plot_segmentation/enhanced_with_plots_morphological.png'

# # Using your updated Tile Size of 256
# TILE_SIZE = 256
# GRID_WIDTH = 14 

# def get_grid_offsets(file_name):
#     """Calculates X and Y pixel offsets based on the tile filename."""
#     try:
#         # Extracts digits from 'tile_0001.png'
#         tile_idx = int(''.join(filter(str.isdigit, file_name)))
#         row = tile_idx // GRID_WIDTH
#         col = tile_idx % GRID_WIDTH
#         return col * TILE_SIZE, row * TILE_SIZE
#     except:
#         return 0, 0

# def main():
#     # 1. Load the full PNG image
#     print(f"Loading image: {FULL_IMAGE_PATH}...")
#     full_image = cv2.imread(FULL_IMAGE_PATH)
#     if full_image is None:
#         print(f"Error: Could not load the image at {FULL_IMAGE_PATH}. Check if the file exists.")
#         return

#     h_img, w_img = full_image.shape[:2]
    
#     # 2. Calculation: Scaling Factor
#     # Based on TILE_SIZE 256 and GRID_WIDTH 14, expected width is 3584.
#     # If your PNG is 3730px wide, this will scale coordinates up slightly (~1.04x).
#     expected_width = GRID_WIDTH * TILE_SIZE
#     scale_factor = w_img / expected_width
#     print(f"Image Size: {w_img}x{h_img} | Scale Factor: {scale_factor:.4f}")

#     # 3. Load COCO annotations
#     coco = COCO(JSON_PATH)
#     all_anns = coco.loadAnns(coco.getAnnIds())
    
#     # Use a single static color for all buildings (BGR format: green/teal)
#     STATIC_COLOR = (255, 250, 100)  # Vibrant green/teal in BGR

#     # Create a separate layer for masks to allow transparency
#     overlay = np.zeros_like(full_image)

#     print(f"Processing {len(all_anns)} annotations...")
#     for ann in all_anns:
#         img_info = coco.loadImgs(ann['image_id'])[0]
#         offset_x, offset_y = get_grid_offsets(img_info['file_name'])
        
#         color = STATIC_COLOR

#         # 4. Handle RLE or Polygon formats automatically
#         mask = coco.annToMask(ann)
        
#         # 5. Extract contours from the mask
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         for cnt in contours:
#             # Transform coordinates: Local -> Global -> Scaled to fit PNG
#             cnt = cnt.astype(np.float32)
#             cnt[:, :, 0] = (cnt[:, :, 0] + offset_x) * scale_factor
#             cnt[:, :, 1] = (cnt[:, :, 1] + offset_y) * scale_factor
#             cnt = cnt.astype(np.int32)

#             # Draw filled buildings on overlay and outlines on the image
#             cv2.drawContours(overlay, [cnt], -1, color, -1)
#             cv2.drawContours(full_image, [cnt], -1, color, 1)

#     # 6. Blend for the final result
#     # alpha 0.6 makes the colors pop against the satellite background
#     cv2.addWeighted(overlay, 0.6, full_image, 0.4, 0, full_image)

#     # 7. Save output as PNG
#     print(f"Saving output to {OUTPUT_PATH}...")
#     os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
#     cv2.imwrite(OUTPUT_PATH, full_image)
#     print("Done!")

# if __name__ == "__main__":
#     main()

import sys
import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO

# --- Configuration ---
FULL_IMAGE_PATH = 'data/annotated_data/full_image.png'
JSON_PATH = 'data/output/unions_buildings.json' 
OUTPUT_PATH = 'data/plot_segmentation/unions_output.png'

def main():
    # 1. Load Image
    print(f"Loading image: {FULL_IMAGE_PATH}...")
    full_image = cv2.imread(FULL_IMAGE_PATH)
    if full_image is None:
        print(f"Error: Could not load image.")
        return

    # 2. Load JSON
    print(f"Loading JSON: {JSON_PATH}...")
    coco = COCO(JSON_PATH)
    all_anns = coco.loadAnns(coco.getAnnIds())
    print(f"Found {len(all_anns)} buildings.")

    # 3. Setup Visualization
    # Create a copy for the translucent fill
    overlay = full_image.copy()
    
    # Color: Bright Yellow/Green (BGR)
    FILL_COLOR = (255, 250, 100)  
    BORDER_COLOR = (0, 100, 255) # Orange-ish border

    drawn_count = 0

    # 4. Draw DIRECTLY from coordinates (Bypassing annToMask)
    for ann in all_anns:
        if 'segmentation' not in ann: continue
        
        # COCO segmentation is a list of polygons: [[x1, y1, x2, y2, ...], [x1, y1...]]
        for seg in ann['segmentation']:
            # Reshape simple list to (N, 1, 2) array for OpenCV
            # We convert floats to 32-bit integers
            poly = np.array(seg, dtype=np.float32).reshape((-1, 1, 2)).astype(np.int32)
            
            # Draw Solid Fill on Overlay
            cv2.fillPoly(overlay, [poly], FILL_COLOR)
            
            # Draw Border on Main Image (for sharpness)
            cv2.polylines(full_image, [poly], True, BORDER_COLOR, 1)
            
            drawn_count += 1

    print(f"Drawn {drawn_count} polygons.")

    # 5. Blend Overlay and Image
    # alpha=0.6 means 60% original image, 40% colored overlay
    alpha = 0.4
    # cv2.addWeighted(full_image, alpha, overlay, 1 - alpha, 0.4, full_image)
    cv2.addWeighted(overlay, 0.6, full_image, 0.4, 0, full_image)
    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, full_image)
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()