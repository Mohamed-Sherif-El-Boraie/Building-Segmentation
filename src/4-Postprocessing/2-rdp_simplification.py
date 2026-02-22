"""
POST-PROCESSING STEP 2: Polygon Simplification (Ramer-Douglas-Peucker)
Purpose: Simplify jagged/noisy polygon edges while preserving shape accuracy

The Ramer-Douglas-Peucker algorithm:
- Reduces number of vertices in polygon
- Maintains polygon shape within epsilon distance
- Useful for removing high-frequency noise from segmentation boundaries
"""

import json
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rdp_simplification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RDPSimplifier:
    """Apply Ramer-Douglas-Peucker simplification to polygon annotations"""
    
    def __init__(self, epsilon=1.0, min_points=4):
        """
        Args:
            epsilon: Maximum distance between original and simplified curve (in pixels)
                    Larger epsilon = more aggressive simplification
                    0.5-2.0 good for satellite imagery
            min_points: Minimum number of points to keep in polygon
        """
        self.epsilon = epsilon
        self.min_points = min_points
        
        logger.info(f"Initialized RDPSimplifier")
        logger.info(f"Epsilon: {epsilon} pixels (max deviation)")
        logger.info(f"Minimum points: {min_points}")
    
    @staticmethod
    def point_to_line_distance(point, line_start, line_end):
        """
        Calculate perpendicular distance from point to line segment
        
        Args:
            point: (x, y) coordinates
            line_start: (x, y) start of line
            line_end: (x, y) end of line
            
        Returns:
            Distance in pixels
        """
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Handle degenerate case (line_start == line_end)
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_length_sq == 0:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        # Calculate perpendicular distance
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
        
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        distance = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        return distance
    
    def rdp_simplify(self, polygon, epsilon):
        """
        Ramer-Douglas-Peucker algorithm for polygon simplification
        
        Args:
            polygon: List of [x, y] coordinates
            epsilon: Maximum distance threshold
            
        Returns:
            Simplified polygon
        """
        if len(polygon) <= self.min_points:
            return polygon
        
        # Convert to array for easier manipulation
        points = np.array(polygon).reshape(-1, 2)
        
        # Find point with maximum distance from line segment start-end
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(points) - 1):
            distance = self.point_to_line_distance(
                points[i],
                points[0],
                points[-1]
            )
            
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance exceeds epsilon, recursively simplify
        if max_distance > epsilon:
            # Recursively simplify the two segments
            left_segment = self.rdp_simplify(
                points[:max_index + 1].tolist(),
                epsilon
            )
            right_segment = self.rdp_simplify(
                points[max_index:].tolist(),
                epsilon
            )
            
            # Combine segments (avoid duplicate point at junction)
            result = left_segment[:-1] + right_segment
        else:
            # Keep only endpoints
            result = [points[0].tolist(), points[-1].tolist()]
        
        return result
    
    def simplify_polygon(self, polygon):
        """
        Simplify a single polygon
        
        Args:
            polygon: List of [x, y] coordinates
            
        Returns:
            Simplified polygon
        """
        if not polygon or len(polygon) < self.min_points:
            return polygon
        
        try:
            simplified = self.rdp_simplify(polygon, self.epsilon)
            
            # Ensure minimum points retained
            if len(simplified) < self.min_points:
                # If too aggressive, fall back to OpenCV's approxPolyDP
                points = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
                simplified_cv = cv2.approxPolyDP(
                    points,
                    self.epsilon,
                    True
                )
                simplified = simplified_cv.reshape(-1, 2).tolist()
            
            return simplified
            
        except Exception as e:
            logger.warning(f"Error simplifying polygon: {e}, returning original")
            return polygon
    
    def process_annotation(self, annotation):
        """
        Process single annotation with RDP simplification
        
        Args:
            annotation: Single COCO annotation dict
            
        Returns:
            Modified annotation with simplified segmentation
        """
        try:
            # Handle RLE format - skip (can't simplify RLE directly)
            if isinstance(annotation['segmentation'], dict):
                logger.debug(f"Skipping RLE annotation {annotation['id']}, converting from mask would be needed")
                return annotation
            
            # Handle polygon format
            elif isinstance(annotation['segmentation'], list):
                simplified_segmentations = []
                
                for polygon in annotation['segmentation']:
                    if not polygon or len(polygon) < 4:
                        simplified_segmentations.append(polygon)
                        continue
                    
                    # Reshape to pairs of coordinates
                    points = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            points.append([polygon[i], polygon[i + 1]])
                    
                    if len(points) < self.min_points:
                        simplified_segmentations.append(polygon)
                        continue
                    
                    # Simplify
                    simplified_points = self.simplify_polygon(points)
                    
                    # Convert back to flat list format
                    simplified_polygon = []
                    for point in simplified_points:
                        simplified_polygon.extend([point[0], point[1]])
                    
                    simplified_segmentations.append(simplified_polygon)
                
                annotation['segmentation'] = simplified_segmentations
                return annotation
            
            else:
                logger.warning(f"Unknown segmentation format for annotation {annotation['id']}")
                return annotation
        
        except Exception as e:
            logger.error(f"Error processing annotation {annotation['id']}: {e}")
            return annotation
    
    def get_simplification_stats(self, input_annotation):
        """
        Calculate simplification statistics for an annotation
        
        Args:
            input_annotation: Original annotation
            
        Returns:
            Dict with statistics
        """
        stats = {
            'original_points': 0,
            'simplified_points': 0,
            'reduction_percentage': 0.0
        }
        
        try:
            if isinstance(input_annotation.get('segmentation'), list):
                for polygon in input_annotation['segmentation']:
                    if polygon:
                        original_points = len(polygon) // 2
                        stats['original_points'] += original_points
        except:
            pass
        
        return stats
    
    def process_coco_json(self, input_json_path, output_json_path):
        """
        Process entire COCO JSON file with RDP simplification
        
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
        
        logger.info(f"Processing {len(coco_data['annotations'])} annotations...")
        
        total_original_points = 0
        total_simplified_points = 0
        
        processed_annotations = []
        for annotation in tqdm(coco_data['annotations'], desc="RDP Simplification"):
            # Calculate before simplification
            if isinstance(annotation.get('segmentation'), list):
                for polygon in annotation['segmentation']:
                    total_original_points += len(polygon) // 2
            
            # Process annotation
            processed_annotation = self.process_annotation(annotation)
            
            # Calculate after simplification
            if isinstance(processed_annotation.get('segmentation'), list):
                for polygon in processed_annotation['segmentation']:
                    total_simplified_points += len(polygon) // 2
            
            processed_annotations.append(processed_annotation)
        
        # Calculate reduction statistics
        if total_original_points > 0:
            reduction = ((total_original_points - total_simplified_points) / total_original_points) * 100
            logger.info(f"\nSimplification Statistics:")
            logger.info(f"  Original total points: {total_original_points}")
            logger.info(f"  Simplified total points: {total_simplified_points}")
            logger.info(f"  Reduction: {reduction:.1f}%")
        
        # Update COCO data
        coco_data['annotations'] = processed_annotations
        
        # Save processed JSON
        logger.info(f"\nSaving processed JSON to {output_json_path}")
        try:
            with open(output_json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            logger.info("✓ Successfully saved processed JSON")
            return True
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return False


def main():
    """Main execution"""
    
    # Configuration
    INPUT_JSON = "data/output/1_morphological_coco.json"  # Output from step 1
    OUTPUT_JSON = "data/output/2_rdp_simplified_coco_pipeline.json"
    
    # RDP Parameters
    EPSILON = 1.0  # Adjust based on detail level
                   # 0.5: Minimal simplification, more detail
                   # 1.0: Balanced (recommended for satellite)
                   # 2.0: More aggressive simplification
    MIN_POINTS = 4  # Keep at least 4 points (minimum for polygon)
    
    # Create output directory
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize simplifier
    simplifier = RDPSimplifier(
        epsilon=EPSILON,
        min_points=MIN_POINTS
    )
    
    # Process COCO JSON
    success = simplifier.process_coco_json(INPUT_JSON, OUTPUT_JSON)
    
    if success:
        logger.info(f"\n✓ Processing complete! Output saved to {OUTPUT_JSON}")
    else:
        logger.error("✗ Processing failed!")
    
    return success


if __name__ == "__main__":
    main()