import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import cv2
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime

from config.logger import get_logger

logger = get_logger(__name__)

class MorphologicalProcessor:
    """Apply morphological operations to COCO segmentation masks"""
    
    def __init__(self, 
                 kernel_size=5,
                 opening_iterations=1,
                 closing_iterations=1):
        """
        Args:
            kernel_size: Size of morphological kernel (must be odd)
            opening_iterations: Number of opening iterations
            closing_iterations: Number of closing iterations
        """
        self.kernel_size = kernel_size
        self.opening_iterations = opening_iterations
        self.closing_iterations = closing_iterations
        
        # Create morphological kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (kernel_size, kernel_size)
        )
        
        logger.info(f"Initialized MorphologicalProcessor")
        logger.info(f"Kernel size: {kernel_size}x{kernel_size}")
        logger.info(f"Opening iterations: {opening_iterations}")
        logger.info(f"Closing iterations: {closing_iterations}")
    
    @staticmethod
    def polygon_to_mask(polygon, image_height, image_width):
        """
        Convert polygon coordinates to binary mask
        
        Args:
            polygon: List of [x, y] coordinates
            image_height: Height of image
            image_width: Width of image
            
        Returns:
            Binary mask (numpy array)
        """
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        if not polygon or len(polygon) < 3:
            return mask
        
        try:
            # Reshape polygon to format required by fillPoly
            points = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [points], 1)
        except Exception as e:
            logger.warning(f"Error converting polygon to mask: {e}")
        
        return mask
    
    @staticmethod
    def mask_to_polygon(mask):
        """
        Convert binary mask back to polygon coordinates
        
        Args:
            mask: Binary mask (numpy array)
            
        Returns:
            List of [x, y] coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Get largest contour (main object)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert to polygon format [x, y, x, y, ...]
        polygon = largest_contour.flatten().tolist()
        
        return polygon
    
    def apply_opening(self, mask):
        """
        Apply morphological opening (erosion + dilation)
        Removes small objects and noise
        """
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=self.opening_iterations
        )
        return opened
    
    def apply_closing(self, mask):
        """
        Apply morphological closing (dilation + erosion)
        Fills small holes in objects
        """
        closed = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.closing_iterations
        )
        return closed
    
    def process_annotation(self, annotation, image_height, image_width):
        """
        Process single annotation with morphological operations
        
        Args:
            annotation: Single COCO annotation dict
            image_height: Height of image
            image_width: Width of image
            
        Returns:
            Modified annotation with processed segmentation
        """
        try:
            # Handle RLE format
            if isinstance(annotation['segmentation'], dict):
                # RLE format - convert to mask
                from pycocotools import mask as mask_utils
                rle = annotation['segmentation']
                mask = mask_utils.decode(rle).astype(np.uint8)
            
            # Handle polygon format
            elif isinstance(annotation['segmentation'], list):
                # Polygon format
                polygon = annotation['segmentation'][0]
                mask = self.polygon_to_mask(
                    polygon,
                    image_height,
                    image_width
                )
            else:
                logger.warning(f"Unknown segmentation format for annotation {annotation['id']}")
                return annotation
            
            # Apply morphological operations
            # First opening to remove noise
            processed_mask = self.apply_opening(mask)
            
            # Then closing to fill holes
            processed_mask = self.apply_closing(processed_mask)
            
            # Convert back to polygon format
            processed_polygon = self.mask_to_polygon(processed_mask)
            
            if processed_polygon:
                annotation['segmentation'] = [processed_polygon]
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error processing annotation {annotation['id']}: {e}")
            return annotation
    
    def process_coco_json(self, input_json_path, output_json_path):
        """
        Process entire COCO JSON file with morphological operations
        
        Args:
            input_json_path: Path to input COCO JSON
            output_json_path: Path to save processed JSON
        """
        logger.info(f"Loading COCO JSON from {input_json_path}")
        
        try:
            with open(input_json_path, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return False
        
        # Create image lookup for dimensions
        image_lookup = {img['id']: img for img in coco_data['images']}
        
        logger.info(f"Processing {len(coco_data['annotations'])} annotations...")
        
        processed_annotations = []
        for annotation in tqdm(coco_data['annotations'], desc="Morphological Operations"):
            image_id = annotation['image_id']
            image = image_lookup.get(image_id)
            
            if not image:
                logger.warning(f"Image {image_id} not found in metadata")
                processed_annotations.append(annotation)
                continue
            
            image_height = image['height']
            image_width = image['width']
            
            # Process annotation
            processed_annotation = self.process_annotation(
                annotation,
                image_height,
                image_width
            )
            processed_annotations.append(processed_annotation)
        
        # Update COCO data
        coco_data['annotations'] = processed_annotations
        
        # Save processed JSON
        logger.info(f"Saving processed JSON to {output_json_path}")
        try:
            with open(output_json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            logger.info("✓ Successfully saved processed JSON")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False
    
    def get_statistics(self, input_json_path):
        """Print statistics about the dataset"""
        with open(input_json_path, 'r') as f:
            coco_data = json.load(f)
        
        logger.info("\n" + "="*50)
        logger.info("DATASET STATISTICS")
        logger.info("="*50)
        logger.info(f"Total images: {len(coco_data['images'])}")
        logger.info(f"Total annotations: {len(coco_data['annotations'])}")
        logger.info(f"Total categories: {len(coco_data['categories'])}")
        
        # Count annotations per category
        category_counts = {}
        for ann in coco_data['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        for cat_id, count in category_counts.items():
            cat_name = next(
                (c['name'] for c in coco_data['categories'] if c['id'] == cat_id),
                f"Unknown (ID: {cat_id})"
            )
            logger.info(f"  {cat_name}: {count} annotations")
        logger.info("="*50 + "\n")


def main():
    """Main execution"""
    
    # Configuration
    INPUT_JSON = "data/annotated_data/merged_buildings.json"  # Change this to your JSON path
    OUTPUT_JSON = "data/output/1_morphological_coco.json"
    
    # Morphological parameters
    KERNEL_SIZE = 5  # Adjust based on object size (must be odd)
    OPENING_ITERATIONS = 1
    CLOSING_ITERATIONS = 1
    
    # Create output directory
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = MorphologicalProcessor(
        kernel_size=KERNEL_SIZE,
        opening_iterations=OPENING_ITERATIONS,
        closing_iterations=CLOSING_ITERATIONS
    )
    
    # Print statistics
    processor.get_statistics(INPUT_JSON)
    
    # Process COCO JSON
    success = processor.process_coco_json(INPUT_JSON, OUTPUT_JSON)
    
    if success:
        logger.info(f"\n✓ Processing complete! Output saved to {OUTPUT_JSON}")
    else:
        logger.error("✗ Processing failed!")
    
    return success


if __name__ == "__main__":
    main()