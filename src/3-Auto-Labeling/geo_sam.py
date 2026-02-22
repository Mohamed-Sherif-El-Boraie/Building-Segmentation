import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
from samgeo import SamGeo3

from config.config import *
from config.logger import get_logger

logger = get_logger("geo_sam")

# ── Paths ──────────────────────────────────────────────────────────────
image_path = "data/test/enhanced.png"                         # local GeoTIFF image
checkpoint = SEGMENTATION_MODEL                 # local model weights  ("models/sam3.pt")

# Output paths
output_masks          = str(Path(SEGMENTATION_OUTPUT_DIR) / "building_masks.tif")
output_masks_binary   = str(Path(SEGMENTATION_OUTPUT_DIR) / "building_masks_binary.tif")
output_masks_scores   = str(Path(SEGMENTATION_OUTPUT_DIR) / "building_masks_with_scores.tif")
output_scores         = str(Path(SEGMENTATION_OUTPUT_DIR) / "building_scores.tif")

os.makedirs(SEGMENTATION_OUTPUT_DIR, exist_ok=True)

# ── 1. Initialize SAM3 ────────────────────────────────────────────────
# backend  : "meta" (official Meta implementation) or "transformers"
# device   : None = auto-detect (CUDA if available, else CPU)
# checkpoint_path : path to local .pt weights so it won't download from HF
# load_from_HF    : False because we already have the model locally
logger.info(f"Initializing SamGeo3 with checkpoint: {checkpoint}")
sam3 = SamGeo3(
    backend="meta",
    device="cuda",
    checkpoint_path=checkpoint,
    load_from_HF=False,
)

# ── 2. Set the image ──────────────────────────────────────────────────
logger.info(f"Setting image: {image_path}")
sam3.set_image(image_path)

# ── 3. Generate masks with a text prompt ──────────────────────────────
text_prompt = "building"
logger.info(f"Generating masks with text prompt: '{text_prompt}'")
sam3.generate_masks(prompt=text_prompt)

# ── 4. Show results (optional – works in Jupyter / interactive env) ──
# sam3.show_anns()
# sam3.show_masks()

# ── 5. Save masks ─────────────────────────────────────────────────────
# Unique masks – each detected object gets a distinct integer label
logger.info(f"Saving unique masks to: {output_masks}")
sam3.save_masks(output=output_masks, unique=True)

# Binary masks – all foreground pixels become 255
logger.info(f"Saving binary masks to: {output_masks_binary}")
sam3.save_masks(output=output_masks_binary, unique=False)

# ── 6. Save masks with confidence scores ──────────────────────────────
logger.info(f"Saving masks + confidence scores to: {output_masks_scores}")
sam3.save_masks(
    output=output_masks_scores,
    save_scores=output_scores,
    unique=True,
)

logger.info("Done – all outputs saved to: " + SEGMENTATION_OUTPUT_DIR)