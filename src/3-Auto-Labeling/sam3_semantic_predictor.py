import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
from datetime import datetime
import cv2

import rasterio
from ultralytics.models.sam import SAM3SemanticPredictor

from config.config import *
from config.logger import get_logger

logger = get_logger("sam3_semantic_predictor")


def mask_to_polygon(mask):
    """Convert binary mask to polygon points (COCO format)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Flatten contour and convert to list
        if len(contour) >= 3:  # Valid polygon needs at least 3 points
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:  # At least 3 x,y pairs
                polygons.append(polygon)
    return polygons


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x, y, width, height]."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def calculate_area(mask):
    """Calculate area of the mask."""
    return int(np.sum(mask))


def create_coco_annotation(mask, annotation_id, image_id, category_id):
    """Create a single COCO annotation from a mask."""
    polygons = mask_to_polygon(mask)
    bbox = mask_to_bbox(mask)
    area = calculate_area(mask)
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": polygons,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }


def results_to_coco(results, image_path, image_id, categories, annotation_id_start=1):
    """
    Convert SAM3 results to COCO format annotations.
    
    Returns:
        - image_info: dict with image metadata
        - annotations: list of annotation dicts
        - next_annotation_id: int for continuing annotation numbering
    """
    # Get image dimensions
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    
    image_info = {
        "id": image_id,
        "file_name": Path(image_path).name,
        "width": width,
        "height": height
    }
    
    annotations = []
    annotation_id = annotation_id_start
    
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        
        for i, mask in enumerate(masks):
            # Determine category based on mask index or class prediction
            category_id = 1  # Default category; adjust based on your categories
            
            # If results have class predictions
            if hasattr(results, 'boxes') and results.boxes is not None:
                if hasattr(results.boxes, 'cls') and len(results.boxes.cls) > i:
                    category_id = int(results.boxes.cls[i].item()) + 1
            
            annotation = create_coco_annotation(
                mask, 
                annotation_id, 
                image_id, 
                category_id
            )
            
            if annotation["segmentation"]:  # Only add if valid polygons exist
                annotations.append(annotation)
                annotation_id += 1
    
    return image_info, annotations, annotation_id


def save_coco_json(coco_data, output_path):
    """Save COCO format data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    logger.info(f"Saved COCO annotations to {output_path}")


def save_geotiff_overlay(image_path, results, output_dir, blue=(0, 20, 230), alpha=0.6):
    """Save segmentation overlay as GeoTIFF preserving all bands + metadata."""
    suffix = Path(image_path).suffix.lower()
    is_tif = suffix in (".tif", ".tiff")

    output_name = Path(image_path).stem + "_overlay"

    if is_tif:
        # Preserve all bands + geo metadata
        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            all_bands = src.read()

        rgb = all_bands[:3].transpose(1, 2, 0).copy()
        mask_layer = np.zeros_like(rgb)

        if results.masks is not None:
            for mask in results.masks.data.cpu().numpy():
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if len(c) >= 3:
                        cv2.fillPoly(mask_layer, [c], blue)

        mask_binary = mask_layer.any(axis=2)
        rgb[mask_binary] = (
            rgb[mask_binary] * (1 - alpha) + mask_layer[mask_binary] * alpha
        ).astype(rgb.dtype)

        all_bands[:3] = rgb.transpose(2, 0, 1)

        output_path = str(Path(output_dir) / f"{output_name}.tif")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(all_bands)
    else:
        # Regular image — just save annotated PNG
        output_path = str(Path(output_dir) / f"{output_name}.png")
        annotated = results.plot(boxes=False, labels=False, conf=False)
        cv2.imwrite(output_path, annotated)

    logger.info(f"Saved overlay to {output_path}")


# Initialize predictor with configuration
overrides = dict(
    conf=0.35,
    task="segment",
    mode="predict",
    model=SEGMENTATION_MODEL,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Define categories (customize based on your use case)
categories = [
    {"id": 1, "name": "building", "supercategory": "structure"},
]

# Initialize COCO structure
coco_output = {
    "info": {
        "description": "SAM3 Semantic Segmentation Output",
        "version": "1.0",
        "year": datetime.now().year,
        "date_created": datetime.now().isoformat()
    },
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": categories
}

image_id = 1
annotation_id = 1

# Get list of images
image_files = list(Path(SMALL_IMAGES_DIR).glob("*.tif")) + \
              list(Path(SMALL_IMAGES_DIR).glob("*.tiff"))

for image_path in image_files:
    logger.info(f"Processing: {image_path}")
    
    predictor.set_image(str(image_path))
    
    # Query with text prompts
    results = predictor(text=["Square Buildings or Rectangular Buildings or Odd-shaped Buildings"])[0]  # Get first result
    
    # Convert results to COCO format
    image_info, annotations, annotation_id = results_to_coco(
        results,
        image_path,
        image_id,
        categories,
        annotation_id
    )
    
    coco_output["images"].append(image_info)
    coco_output["annotations"].extend(annotations)
    
    # Optionally save visualization
    save_geotiff_overlay(image_path, results, SEGMENTATION_OUTPUT_DIR)
    
    image_id += 1

# Save COCO JSON output
output_json_path = Path(SEGMENTATION_OUTPUT_DIR) / "annotations.json"
save_coco_json(coco_output, output_json_path)

logger.info(f"Processed {len(coco_output['images'])} images")
logger.info(f"Generated {len(coco_output['annotations'])} annotations")