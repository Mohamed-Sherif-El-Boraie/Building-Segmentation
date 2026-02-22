import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import cv2
import rasterio
from pathlib import Path
from typing import Tuple, Optional

from config.config import *
from config.logger import get_logger

logger = get_logger("plot_tif")

def load_coco_annotations(json_path: str) -> dict:
    """Load COCO format annotations from JSON file."""
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def load_tiff_all_bands(tiff_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load ALL bands from TIFF image.
    
    Returns:
        all_bands: numpy array of shape (bands, height, width)
        metadata: rasterio metadata dict (CRS, transform, etc.)
    """
    with rasterio.open(tiff_path) as src:
        # Store metadata
        metadata = src.meta.copy()
        
        # Read ALL bands (shape: bands, height, width)
        all_bands = src.read()
        
        return all_bands, metadata


def save_tiff_all_bands(all_bands: np.ndarray, output_path: str, metadata: dict):
    """
    Save ALL bands as GeoTIFF preserving metadata.
    """
    # Update metadata for output
    out_meta = metadata.copy()
    out_meta.update({
        'driver': 'GTiff',
        'count': all_bands.shape[0],
        'height': all_bands.shape[1],
        'width': all_bands.shape[2],
        'compress': 'lzw'
    })
    
    # Write the TIFF
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(all_bands)
    
    logger.info(f"Saved GeoTIFF:")
    logger.info(f"   - Bands: {all_bands.shape[0]}")
    logger.info(f"   - Size: {all_bands.shape[2]}x{all_bands.shape[1]}")
    logger.info(f"   - CRS: {out_meta.get('crs', 'None')}")


def draw_segmentation_on_rgb(
    rgb_bands: np.ndarray,
    coco_data: dict,
    outline_color: Tuple[int, int, int] = (0, 255, 0),  # RGB green
    fill_color: Optional[Tuple[int, int, int]] = (0, 255, 0),
    outline_thickness: int = 2,
    fill_alpha: float = 0.25
) -> np.ndarray:
    """
    Draw segmentation polygons on RGB bands using OpenCV.
    
    Args:
        rgb_bands: RGB image of shape (3, height, width) or (height, width, 3)
        coco_data: COCO format annotation dictionary
        outline_color: RGB color for polygon outlines
        fill_color: RGB color for polygon fill (None for outline only)
        outline_thickness: Thickness of outline in pixels
        fill_alpha: Transparency of fill (0.0 - 1.0)
    
    Returns:
        Annotated RGB bands in same format as input
    """
    # Check input format and convert to (H, W, 3) for OpenCV
    if rgb_bands.shape[0] == 3:
        # Input is (3, H, W) - rasterio format
        image = np.transpose(rgb_bands, (1, 2, 0)).copy()
        input_format = 'bands_first'
    else:
        # Input is (H, W, 3)
        image = rgb_bands.copy()
        input_format = 'bands_last'
    
    # Ensure uint8 for OpenCV
    if image.dtype != np.uint8:
        # Normalize to 0-255
        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    
    # Create annotated image
    annotated = image.copy()
    
    # Create overlay for transparent fill
    if fill_color is not None:
        overlay = image.copy()
    
    annotations = coco_data.get('annotations', [])
    logger.info(f"Drawing {len(annotations)} annotations...")
    
    drawn_count = 0
    
    for ann in annotations:
        segmentation = ann.get('segmentation', [])
        
        # Handle polygon format
        if isinstance(segmentation, list) and len(segmentation) > 0:
            for seg in segmentation:
                if isinstance(seg, list) and len(seg) >= 6:
                    # Convert flat list [x1,y1,x2,y2,...] to points array
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    
                    # OpenCV uses BGR, but we're working in RGB here
                    # Convert RGB to BGR for OpenCV operations
                    outline_bgr = (outline_color[2], outline_color[1], outline_color[0])
                    fill_bgr = (fill_color[2], fill_color[1], fill_color[0]) if fill_color else None
                    
                    # Draw filled polygon on overlay
                    if fill_bgr is not None:
                        cv2.fillPoly(overlay, [points], fill_bgr)
                    
                    # Draw outline on annotated image
                    if outline_thickness > 0:
                        cv2.polylines(annotated, [points], isClosed=True, 
                                     color=outline_bgr, thickness=outline_thickness)
                    
                    drawn_count += 1
        
        # Handle RLE format
        elif isinstance(segmentation, dict) and 'counts' in segmentation:
            logger.info("RLE segmentation detected - skipping (requires pycocotools)")
            continue
    
    # Blend overlay with annotated image for transparent fill
    if fill_color is not None:
        cv2.addWeighted(overlay, fill_alpha, annotated, 1 - fill_alpha, 0, annotated)
    
    logger.info(f"Drew {drawn_count} polygons")
    
    # Convert back to original format
    if input_format == 'bands_first':
        # Convert (H, W, 3) back to (3, H, W)
        annotated = np.transpose(annotated, (2, 0, 1))
    
    return annotated


def plot_annotations(
    tiff_path: str,
    json_path: str,
    output_path: str,
    outline_color: Tuple[int, int, int] = (0, 255, 0),
    fill_color: Optional[Tuple[int, int, int]] = (0, 255, 0),
    outline_thickness: int = 2,
    fill_alpha: float = 0.25,
    rgb_bands: Tuple[int, int, int] = (1, 2, 3)  # Which bands to use as RGB
):
    """
    Main function: Load TIFF, draw annotations on RGB, save ALL bands as GeoTIFF.
    
    Args:
        tiff_path: Path to input TIFF image
        json_path: Path to COCO annotations JSON
        output_path: Where to save the annotated TIFF
        outline_color: RGB color for outlines (default: green)
        fill_color: RGB color for fill (None for outline only)
        outline_thickness: Thickness of outline
        fill_alpha: Transparency of fill (0.0 - 1.0)
        rgb_bands: Which bands to treat as RGB (1-indexed, default: 1,2,3)
    """
    logger.info("PLOT ANNOTATIONS ON TIFF (Preserve All Bands)")
    
    # Check if files exist
    if not Path(json_path).exists():
        logger.error(f"❌ JSON file not found: {json_path}")
        return
    
    if not Path(tiff_path).exists():
        logger.error(f"❌ TIFF file not found: {tiff_path}")
        return
    
    # Step 1: Load ALL bands from TIFF
    logger.info(f"\nStep 1: Loading ALL bands from: {tiff_path}")
    all_bands, metadata = load_tiff_all_bands(tiff_path)
    num_bands = all_bands.shape[0]
    logger.info(f"Total bands: {num_bands}")
    logger.info(f"Shape: {all_bands.shape} (bands, height, width)")
    logger.info(f"Dtype: {all_bands.dtype}")
    logger.info(f"CRS: {metadata.get('crs', 'None')}")
    
    # Step 2: Load annotations
    logger.info(f"Step 2: Loading annotations from: {json_path}")
    coco_data = load_coco_annotations(json_path)
    logger.info(f"Annotations: {len(coco_data.get('annotations', []))}")
    
    # Step 3: Extract RGB bands for annotation
    logger.info(f"Step 3: Drawing annotations on bands {rgb_bands}...")
    
    # Convert 1-indexed to 0-indexed
    r_idx, g_idx, b_idx = rgb_bands[0] - 1, rgb_bands[1] - 1, rgb_bands[2] - 1
    
    # Check if bands exist
    if max(r_idx, g_idx, b_idx) >= num_bands:
        logger.error(f"Error: RGB bands {rgb_bands} exceed available bands ({num_bands})")
        return
    
    # Extract RGB bands (shape: 3, H, W)
    rgb = np.stack([
        all_bands[r_idx],
        all_bands[g_idx],
        all_bands[b_idx]
    ], axis=0)
    
    # Draw annotations on RGB
    rgb_annotated = draw_segmentation_on_rgb(
        rgb_bands=rgb,
        coco_data=coco_data,
        outline_color=outline_color,
        fill_color=fill_color,
        outline_thickness=outline_thickness,
        fill_alpha=fill_alpha
    )
    
    # Step 4: Replace RGB bands in all_bands with annotated version
    logger.info(f"Step 4: Updating bands {rgb_bands} with annotations...")
    
    # Create output array (copy of original)
    output_bands = all_bands.copy()
    
    # Convert annotated RGB back to original dtype if needed
    if all_bands.dtype != np.uint8:
        # Scale back to original range
        orig_min = all_bands[r_idx].min()
        orig_max = all_bands[r_idx].max()
        rgb_annotated = (rgb_annotated.astype(np.float32) / 255.0 * (orig_max - orig_min) + orig_min).astype(all_bands.dtype)
    
    # Replace bands
    output_bands[r_idx] = rgb_annotated[0]
    output_bands[g_idx] = rgb_annotated[1]
    output_bands[b_idx] = rgb_annotated[2]
    
    logger.info(f"Bands {rgb_bands} updated with annotations")
    logger.info(f"Bands unchanged: {[i+1 for i in range(num_bands) if i not in [r_idx, g_idx, b_idx]]}")
    
    # Step 5: Save ALL bands as GeoTIFF
    logger.info(f"Step 5: Saving GeoTIFF to: {output_path}")
    save_tiff_all_bands(output_bands, output_path, metadata)
    
    # Verify output
    with rasterio.open(output_path) as src:
        out_bands = src.count
        out_crs = src.crs
        out_dtype = src.dtypes[0]
    
    logger.info("COMPLETE!")
    logger.info(f"Input:  {tiff_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Bands:  {num_bands} → {out_bands} (preserved)")
    logger.info(f"Dtype:  {out_dtype}")
    logger.info(f"CRS:    {out_crs}")
    logger.info(f"Modified bands: {list(rgb_bands)} (RGB with annotations)")
    logger.info(f"Unchanged bands: {[i+1 for i in range(num_bands) if i not in [r_idx, g_idx, b_idx]]}")
    logger.info(f"Buildings: {len(coco_data.get('annotations', []))}")


def main():
    # Define paths
    json_path = "data/segmentation_output_2/result_0.json"
    tiff_path = "data/tile_0008_0029.tif"
    output_path = "data/tile_0008_0029_annotated.tif"
    
    # Plot annotations
    plot_annotations(
        tiff_path=tiff_path,
        json_path=json_path,
        output_path=output_path,
        outline_color=(0, 255, 0),    
        fill_color=(0, 255, 0),       
        outline_thickness=2,
        fill_alpha=0.25,
        rgb_bands=(1, 2, 3)         
    )


if __name__ == "__main__":
    main()