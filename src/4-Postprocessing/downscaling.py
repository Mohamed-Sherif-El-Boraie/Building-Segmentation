import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
from typing import Dict, List, Any

from config.config  import *
from config.logger import get_logger

logger = get_logger("downscaling")

def downscale_coco_annotations(
    coco_data: Dict,
    scale_factor: float = 2.0
) -> Dict:
    """
    Downscale all coordinates in COCO annotations.
    """
    # Create a copy to avoid modifying original
    scaled_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Scale image dimensions
    for img in coco_data.get("images", []):
        scaled_img = img.copy()
        scaled_img["width"] = int(img.get("width", 0) / scale_factor)
        scaled_img["height"] = int(img.get("height", 0) / scale_factor)
        scaled_data["images"].append(scaled_img)
    
    # Scale annotations
    for ann in coco_data.get("annotations", []):
        scaled_ann = {
            "id": ann.get("id"),
            "image_id": ann.get("image_id"),
            "category_id": ann.get("category_id"),
            "iscrowd": ann.get("iscrowd", 0)
        }
        
        # Scale segmentation coordinates
        if "segmentation" in ann:
            scaled_segmentation = []
            for seg in ann["segmentation"]:
                # Divide each coordinate by scale_factor
                scaled_seg = [coord / scale_factor for coord in seg]
                scaled_segmentation.append(scaled_seg)
            scaled_ann["segmentation"] = scaled_segmentation
        
        # Scale bounding box [x, y, width, height]
        if "bbox" in ann:
            bbox = ann["bbox"]
            scaled_ann["bbox"] = [
                bbox[0] / scale_factor,  # x
                bbox[1] / scale_factor,  # y
                bbox[2] / scale_factor,  # width
                bbox[3] / scale_factor   # height
            ]
        
        # Scale area (area scales by factor^2)
        if "area" in ann:
            scaled_ann["area"] = ann["area"] / (scale_factor ** 2)
        
        scaled_data["annotations"].append(scaled_ann)
    
    # Update description
    if "info" in scaled_data and scaled_data["info"]:
        original_desc = scaled_data["info"].get("description", "")
        scaled_data["info"]["description"] = f"{original_desc} (downscaled {scale_factor}x)"
    
    return scaled_data


def main():
    """Main function."""
    
    # Configuration
    input_path = Path("data/coco_json_annotations/merged_annotations.json")
    output_path = Path("data/coco_json_annotations/merged_annotations_downscaled.json")
    scale_factor = 2.0  # Divide by 2
    
    logger.info(f"Scale factor: {scale_factor}x (divide coordinates by {scale_factor})")
    
    # Load input
    logger.info(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get original dimensions
    if coco_data.get("images"):
        orig_w = coco_data["images"][0].get("width", 0)
        orig_h = coco_data["images"][0].get("height", 0)
        logger.info(f"Original size: {orig_w} x {orig_h}")
        logger.info(f"New size:      {int(orig_w/scale_factor)} x {int(orig_h/scale_factor)}")
    
    num_annotations = len(coco_data.get("annotations", []))
    logger.info(f"Annotations:   {num_annotations}")
    
    # Downscale
    logger.info(f"Downscaling by {scale_factor}x...")
    scaled_data = downscale_coco_annotations(coco_data, scale_factor)
    
    # Save output
    logger.info(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(scaled_data, f, indent=2)
    
    # Show sample comparison
    if coco_data.get("annotations") and scaled_data.get("annotations"):
        orig_ann = coco_data["annotations"][0]
        scaled_ann = scaled_data["annotations"][0]
        
        logger.info(f"Sample comparison (first annotation):")
        logger.info(f"Original bbox: {orig_ann.get('bbox', [])}")
        logger.info(f"Scaled bbox:   {scaled_ann.get('bbox', [])}")
        logger.info(f"Original area: {orig_ann.get('area', 0):.2f}")
        logger.info(f"Scaled area:   {scaled_ann.get('area', 0):.2f}")
    
    return scaled_data


if __name__ == "__main__":
    main()