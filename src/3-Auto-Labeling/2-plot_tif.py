import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import rasterio
import cv2

IMAGE_PATH = "data/tile_0008_0029.tif"
JSON_PATH = "data/segmentation_output_2/result_0.json"
OUTPUT_PATH = "data/segmentation_output_2/segmentation_overlay.tif"
BLUE = (0, 20, 230)  # RGB blue (rasterio uses RGB, not BGR)
ALPHA = 0.6


# --- Load JSON ---
with open(JSON_PATH) as f:
    data = json.load(f)

# --- Read original GeoTIFF preserving all bands + metadata ---
with rasterio.open(IMAGE_PATH) as src:
    profile = src.profile.copy()
    all_bands = src.read()  # (bands, H, W)
    band_count = src.count

# --- Build mask on first 3 bands (RGB) ---
rgb = all_bands[:3].transpose(1, 2, 0).copy()  # (H, W, 3)
mask_layer = np.zeros_like(rgb)
for seg in data["segments"]:
    for polygon in seg["polygons"]:
        pts = np.array(polygon, dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask_layer, [pts], BLUE)

# --- Blend only where mask exists ---
mask_binary = mask_layer.any(axis=2)
rgb[mask_binary] = (
    rgb[mask_binary] * (1 - ALPHA) + mask_layer[mask_binary] * ALPHA
).astype(rgb.dtype)

# --- Replace RGB bands, keep other bands intact ---
all_bands[:3] = rgb.transpose(2, 0, 1)

# --- Save with original metadata (CRS, transform, etc.) ---
with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
    dst.write(all_bands)
