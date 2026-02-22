import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from pycocotools import mask as mask_utils
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid
import cv2

from config.config import *
from config.logger import get_logger

logger = get_logger("coordinate_mapping")

# Configuration
TILES_PER_ROW = 14      
TILE_SIZE = 256
EDGE_THRESHOLD = 2  # pixels from edge to consider as "touching boundary"


def get_tile_position(tile_name: str) -> Tuple[int, int]:
    """Extract row and column from tile filename."""
    import re
    match = re.search(r'tile_(\d+)', tile_name)
    if not match:
        logger.error(f"Cannot parse tile name: {tile_name}")
    
    tile_num = int(match.group(1))
    row = tile_num // TILES_PER_ROW
    col = tile_num % TILES_PER_ROW
    return row, col


def get_global_offset(tile_name: str) -> Tuple[int, int]:
    """Get the global pixel offset for a tile."""
    row, col = get_tile_position(tile_name)
    return col * TILE_SIZE, row * TILE_SIZE


def rle_to_polygon(rle_annotation: Dict, simplify_tolerance: float = 1.0) -> List[Polygon]:
    """Convert RLE segmentation to Shapely Polygon(s)."""
    if isinstance(rle_annotation, dict) and 'counts' in rle_annotation:
        if isinstance(rle_annotation['counts'], str):
            mask = mask_utils.decode(rle_annotation)
        else:
            mask = mask_utils.decode(rle_annotation)
    else:
        return []
    
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            coords = contour.squeeze()
            if len(coords.shape) == 1:
                continue
            if len(coords) >= 3:
                try:
                    poly = Polygon(coords)
                    if simplify_tolerance > 0:
                        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
                    if poly.is_valid and not poly.is_empty and poly.area > 0:
                        polygons.append(poly)
                    elif not poly.is_empty:
                        poly = make_valid(poly)
                        if isinstance(poly, Polygon) and poly.area > 0:
                            polygons.append(poly)
                        elif isinstance(poly, MultiPolygon):
                            for p in poly.geoms:
                                if p.area > 0:
                                    polygons.append(p)
                except Exception:
                    logger.error(f"Failed to process polygon: {contour}")
                    continue
    
    return polygons


def get_touched_edges(polygon: Polygon, tile_size: int = TILE_SIZE, threshold: int = EDGE_THRESHOLD) -> Set[str]:
    """
    Determine which edges of the tile this polygon touches.
    Returns set of: 'left', 'right', 'top', 'bottom'
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    edges = set()
    
    if bounds[0] <= threshold:
        edges.add('left')
    if bounds[1] <= threshold:
        edges.add('top')
    if bounds[2] >= tile_size - threshold:
        edges.add('right')
    if bounds[3] >= tile_size - threshold:
        edges.add('bottom')
    
    return edges


def transform_polygon_to_global(polygon: Polygon, x_offset: int, y_offset: int) -> Polygon:
    """Transform polygon from tile-local to global coordinates."""
    from shapely.affinity import translate
    return translate(polygon, xoff=x_offset, yoff=y_offset)


def polygon_to_coco_segmentation(polygon: Polygon) -> List[List[float]]:
    """Convert Shapely Polygon to COCO polygon format."""
    if isinstance(polygon, MultiPolygon):
        result = []
        for poly in polygon.geoms:
            result.extend(polygon_to_coco_segmentation(poly))
        return result
    
    if polygon.is_empty or polygon.area == 0:
        return []
    
    coords = list(polygon.exterior.coords)
    flat_coords = []
    for x, y in coords[:-1]:
        flat_coords.extend([float(x), float(y)])
    
    return [flat_coords] if flat_coords else []


def get_edge_zone(tile_row: int, tile_col: int, edge: str, buffer: float = 5.0) -> Polygon:
    """
    Get a thin rectangular zone along the specified edge in global coordinates.
    This is used to find polygons that should be merged across this edge.
    """
    x_offset = tile_col * TILE_SIZE
    y_offset = tile_row * TILE_SIZE
    
    if edge == 'right':
        # Right edge of this tile = left edge of tile to the right
        return box(x_offset + TILE_SIZE - buffer, y_offset, 
                   x_offset + TILE_SIZE + buffer, y_offset + TILE_SIZE)
    elif edge == 'bottom':
        # Bottom edge of this tile = top edge of tile below
        return box(x_offset, y_offset + TILE_SIZE - buffer,
                   x_offset + TILE_SIZE, y_offset + TILE_SIZE + buffer)
    elif edge == 'left':
        return box(x_offset - buffer, y_offset,
                   x_offset + buffer, y_offset + TILE_SIZE)
    elif edge == 'top':
        return box(x_offset, y_offset - buffer,
                   x_offset + TILE_SIZE, y_offset + buffer)
    
    return None


def process_coco_annotations(coco_data: Dict) -> Dict:
    """
    Main processing function - EDGE-ONLY MERGING
    """
    print("Step 1: Building image lookup...")
    image_lookup = {img['id']: img['file_name'] for img in coco_data['images']}
    print(f"  Found {len(image_lookup)} images")
    
    print("\nStep 2: Converting annotations and classifying as edge/interior...")
    
    # Two categories of polygons
    interior_polygons = []  # Keep these unchanged
    edge_polygons = []      # These are candidates for merging
    
    total_annotations = len(coco_data['annotations'])
    
    for i, ann in enumerate(coco_data['annotations']):
        if (i + 1) % 1000 == 0:
            print(f"  Processing annotation {i + 1}/{total_annotations}...")
        
        image_id = ann['image_id']
        if image_id not in image_lookup:
            continue
        
        tile_name = image_lookup[image_id]
        
        try:
            x_offset, y_offset = get_global_offset(tile_name)
            tile_row, tile_col = get_tile_position(tile_name)
        except ValueError as e:
            print(f"  Warning: {e}")
            continue
        
        segmentation = ann.get('segmentation', {})
        if not segmentation:
            continue
        
        local_polygons = rle_to_polygon(segmentation)
        
        for poly in local_polygons:
            # Check which edges this polygon touches
            touched_edges = get_touched_edges(poly)
            
            # Transform to global coordinates
            global_poly = transform_polygon_to_global(poly, x_offset, y_offset)
            
            if not global_poly.is_valid or global_poly.area <= 0:
                continue
            
            poly_data = {
                'polygon': global_poly,
                'category_id': ann['category_id'],
                'tile_row': tile_row,
                'tile_col': tile_col,
                'touched_edges': touched_edges
            }
            
            if touched_edges:
                # This polygon touches at least one edge - candidate for merging
                edge_polygons.append(poly_data)
            else:
                # Interior polygon - keep as-is
                interior_polygons.append(poly_data)
    
    logger.info(f"\n  Interior polygons (unchanged): {len(interior_polygons)}")
    logger.info(f"  Edge polygons (merge candidates): {len(edge_polygons)}")
        
    # Build spatial index for edge polygons by their touched edges
    # Key: (tile_row, tile_col, edge) -> list of polygon indices
    edge_index = defaultdict(list)
    for idx, poly_data in enumerate(edge_polygons):
        for edge in poly_data['touched_edges']:
            key = (poly_data['tile_row'], poly_data['tile_col'], edge)
            edge_index[key].append(idx)
    
    # Track which edge polygons have been merged
    merged_set = set()  # indices of polygons that have been merged into something else
    merged_results = []  # resulting merged polygons
    
    # Process each tile boundary (only right and bottom to avoid duplicates)
    processed_boundaries = set()
    
    for idx, poly_data in enumerate(edge_polygons):
        if idx in merged_set:
            continue
        
        tile_row = poly_data['tile_row']
        tile_col = poly_data['tile_col']
        touched_edges = poly_data['touched_edges']
        
        # Collect all polygons that should be merged with this one
        to_merge = {idx}
        to_check = [idx]
        
        while to_check:
            current_idx = to_check.pop()
            current_data = edge_polygons[current_idx]
            current_poly = current_data['polygon']
            c_row = current_data['tile_row']
            c_col = current_data['tile_col']
            c_edges = current_data['touched_edges']
            
            # Check adjacent tiles for matching edge polygons
            adjacent_checks = []
            
            if 'right' in c_edges and c_col < TILES_PER_ROW - 1:
                # Look for polygons touching left edge of tile to the right
                adjacent_checks.append((c_row, c_col + 1, 'left'))
            
            if 'left' in c_edges and c_col > 0:
                # Look for polygons touching right edge of tile to the left
                adjacent_checks.append((c_row, c_col - 1, 'right'))
            
            if 'bottom' in c_edges and c_row < TILES_PER_ROW - 1:
                # Look for polygons touching top edge of tile below
                adjacent_checks.append((c_row + 1, c_col, 'top'))
            
            if 'top' in c_edges and c_row > 0:
                # Look for polygons touching bottom edge of tile above
                adjacent_checks.append((c_row - 1, c_col, 'bottom'))
            
            for adj_row, adj_col, adj_edge in adjacent_checks:
                key = (adj_row, adj_col, adj_edge)
                
                for adj_idx in edge_index.get(key, []):
                    if adj_idx in to_merge:
                        continue
                    
                    adj_poly = edge_polygons[adj_idx]['polygon']
                    
                    # Check if polygons actually touch/overlap at the boundary
                    # Use a small buffer to handle tiny gaps
                    if current_poly.buffer(3).intersects(adj_poly.buffer(3)):
                        to_merge.add(adj_idx)
                        to_check.append(adj_idx)
        
        # Now merge all polygons in to_merge
        if len(to_merge) > 1:
            # Multiple polygons to merge
            polys_to_merge = [edge_polygons[i]['polygon'] for i in to_merge]
            
            # Buffer slightly, union, then unbuffer to merge cleanly
            buffered = [p.buffer(2) for p in polys_to_merge]
            merged = unary_union(buffered)
            merged = merged.buffer(-2)
            
            if isinstance(merged, Polygon) and merged.area > 0:
                merged_results.append({
                    'polygon': merged,
                    'category_id': poly_data['category_id']
                })
            elif isinstance(merged, MultiPolygon):
                for p in merged.geoms:
                    if p.area > 0:
                        merged_results.append({
                            'polygon': p,
                            'category_id': poly_data['category_id']
                        })
            
            merged_set.update(to_merge)
        else:
            # Single polygon that touches edge but has no match - keep as-is
            merged_results.append({
                'polygon': poly_data['polygon'],
                'category_id': poly_data['category_id']
            })
            merged_set.add(idx)
    
    logger.info(f"  Edge polygons merged into: {len(merged_results)} polygons")
    
    # Combine interior (unchanged) + merged edge polygons
    all_final_polygons = []
    
    # Add interior polygons
    for poly_data in interior_polygons:
        all_final_polygons.append({
            'polygon': poly_data['polygon'],
            'category_id': poly_data['category_id']
        })
    
    # Add merged edge polygons
    all_final_polygons.extend(merged_results)
    
    logger.info(f"  Total buildings: {len(all_final_polygons)}")
    
    # Create output COCO structure
    output_coco = {
        'info': {
            **coco_data.get('info', {}),
            'description': 'Merged building segmentations (edge-only merging)'
        },
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data.get('categories', []),
        'images': [{
            'id': 0,
            'file_name': 'merged_image.jpg',
            'width': TILES_PER_ROW * TILE_SIZE,
            'height': TILES_PER_ROW * TILE_SIZE
        }],
        'annotations': []
    }
    
    for idx, item in enumerate(all_final_polygons):
        poly = item['polygon']
        
        if not poly.is_valid:
            poly = make_valid(poly)
        
        if isinstance(poly, MultiPolygon):
            # Handle MultiPolygon - add each part separately
            for sub_poly in poly.geoms:
                if sub_poly.area > 0:
                    bounds = sub_poly.bounds
                    segmentation = polygon_to_coco_segmentation(sub_poly)
                    if segmentation:
                        output_coco['annotations'].append({
                            'id': len(output_coco['annotations']),
                            'image_id': 0,
                            'category_id': item['category_id'],
                            'segmentation': segmentation,
                            'bbox': [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]],
                            'area': sub_poly.area,
                            'iscrowd': 0
                        })
        elif poly.area > 0:
            bounds = poly.bounds
            segmentation = polygon_to_coco_segmentation(poly)
            if segmentation:
                output_coco['annotations'].append({
                    'id': len(output_coco['annotations']),
                    'image_id': 0,
                    'category_id': item['category_id'],
                    'segmentation': segmentation,
                    'bbox': [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]],
                    'area': poly.area,
                    'iscrowd': 0
                })
    
    return output_coco


def main():
    input_json = Path("data/annotated_data/train/_annotations.coco.json")
    output_json = Path("data/coco_json_annotations/merged_annotations_256.json")
    

    logger.info(f"Grid: {TILES_PER_ROW}x{TILES_PER_ROW} tiles")
    logger.info(f"Tile size: {TILE_SIZE}x{TILE_SIZE} pixels")
    logger.info(f"Edge threshold: {EDGE_THRESHOLD} pixels")

    
    logger.info(f"\nLoading input: {input_json}")
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    logger.info(f"Images: {len(coco_data['images'])}")
    logger.info(f"Annotations: {len(coco_data['annotations'])}")
    
    merged_coco = process_coco_annotations(coco_data)
    
    logger.info(f"\nSaving merged annotations to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(merged_coco, f, indent=2)
    
    logger.info("SUMMARY")
    logger.info(f"Input annotations: {len(coco_data['annotations'])}")
    logger.info(f"Output annotations: {len(merged_coco['annotations'])}")
    logger.info(f"Net change: {len(coco_data['annotations']) - len(merged_coco['annotations'])} fewer (merged at edges)")
    
    return merged_coco


if __name__ == "__main__":
    main()