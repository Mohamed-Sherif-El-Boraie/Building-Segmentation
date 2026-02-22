import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import cv2
import json
import numpy as np

import rasterio
from ultralytics.models.sam import SAM3SemanticPredictor

from config.config import *
from config.logger import get_logger

logger = get_logger("sam3_semantic_predictor")

# Initialize predictor with configuration
overrides = dict(
    conf=0.35,
    task="segment",
    mode="predict",
    model=SEGMENTATION_MODEL,
    # half=True,  # Use FP16 for faster inference
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries

image = "data/test/enhanced.png"

predictor.set_image(image)
# Query with multiple text prompts
# results = predictor(text=["Buildings"], save=True)

# Works with descriptive phrases
# results = predictor(text=["Square rooftops or Rectangular roof structures or Odd-shaped roof structures"], save=True)

# Query with a single concept
results = predictor(text=["Square Buildings or Rectangular Buildings or Odd-shaped Buildings"], bboxes=None, labels=None)

os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)


# Export to JSON
for i, result in enumerate(results):
    # Save annotated image
    save_path = str(Path(SEGMENTATION_OUTPUT_DIR) / f"result_{i}.png")
    annotated = result.plot(boxes=False, labels=False, conf=False)
    cv2.imwrite(save_path, annotated)

    # Build JSON output
    segments = []
    if result.masks is not None and result.boxes is not None:
        for j, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
            # Convert binary mask to polygon contours
            mask_np = mask.cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [c.squeeze().tolist() for c in contours if len(c) >= 3]

            segments.append({
                "id": j,
                "class": result.names[int(box.cls)] if int(box.cls) < len(result.names) else "unknown",
                "confidence": round(float(box.conf), 4),
                "bbox_xyxy": box.xyxy[0].cpu().tolist(),
                "polygons": polygons,
            })

    output = {
        "image": str(Path(image).name),
        "image_width": result.orig_img.shape[1],
        "image_height": result.orig_img.shape[0],
        "num_detections": len(segments),
        "segments": segments,
    }

    json_path = str(Path(SEGMENTATION_OUTPUT_DIR) / f"result_{i}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(segments)} segments to {json_path}")

