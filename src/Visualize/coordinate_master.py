import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# --- GEOMETRY IMPORTS ---
from pycocotools import mask as mask_utils
from shapely.geometry import Polygon, MultiPolygon, box, GeometryCollection
from shapely.ops import unary_union
from shapely.validation import make_valid
import cv2

# --- LOGGER SETUP (Simplified for standalone use) ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coordinate_mapping")

# --- CONFIGURATION ---
TILES_PER_ROW = 14      
TILE_SIZE = 256
# AGGRESSIVE SETTINGS
EDGE_THRESHOLD = 10  # Look for buildings within 10px of edge (was 2)
MERGE_BUFFER = 5     # Expand building by 5px to find neighbors (bridges gaps)

def get_tile_position(tile_name: str) -> Tuple[int, int]:
    """Extract row and column from tile filename (e.g., tile_0015.png)."""
    import re
    # Matches any number in the string
    match = re.search(r'(\d+)', tile_name)
    if not match:
        # Fallback if naming is different, assuming 0
        return 0, 0
    
    tile_num = int(match.group(1))
    row = tile_num // TILES_PER_ROW
    col = tile_num % TILES_PER_ROW
    return row, col

def get_global_offset(tile_name: str) -> Tuple[int, int]:
    row, col = get_tile_position(tile_name)
    return col * TILE_SIZE, row * TILE_SIZE

def rle_to_polygon(rle_annotation: Dict) -> List[Polygon]:
    """Convert RLE or Poly segmentation to Shapely Polygon(s)."""
    if isinstance(rle_annotation, list):
        # Already polygon format
        rle = mask_utils.frPyObjects(rle_annotation, TILE_SIZE, TILE_SIZE)
        mask = mask_utils.decode(rle)
        # Flatten if needed
        if len(mask.shape) == 3: mask = mask[:, :, 0]
    elif isinstance(rle_annotation, dict):
        mask = mask_utils.decode(rle_annotation)
    else:
        return []
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if len(contour) >= 3:
            poly = Polygon(contour.squeeze())
            if not poly.is_valid:
                poly = make_valid(poly)
                
            if isinstance(poly, Polygon) and poly.area > 1:
                polygons.append(poly)
            elif isinstance(poly, MultiPolygon):
                for p in poly.geoms:
                    if p.area > 1: polygons.append(p)
    return polygons

def get_touched_edges(polygon: Polygon) -> Set[str]:
    """Determine which edges (Global) the polygon is near."""
    # Since we transform to global BEFORE calling this, we need logic relative to the tile
    # BUT easier approach: The logic in main handles the categorization. 
    # Here we just check bounds relative to the TILE SIZE (0-256)
    pass 

def polygon_to_coco(polygon):
    """Robust export to COCO format."""
    if polygon.is_empty: return []

    valid_polys = []
    
    def extract(geom):
        if isinstance(geom, Polygon): valid_polys.append(geom)
        elif isinstance(geom, MultiPolygon): valid_polys.extend(geom.geoms)
        elif isinstance(geom, GeometryCollection):
            for g in geom.geoms: extract(g)

    extract(polygon)
    
    results = []
    for p in valid_polys:
        if hasattr(p, 'exterior') and p.exterior:
            coords = list(p.exterior.coords)
            flat = [round(x, 2) for pt in coords for x in pt]
            if len(flat) >= 6: results.append(flat)
    return results

def process_coco_annotations(coco_data: Dict) -> Dict:
    print("Step 1: Mapping all polygons to Global Coordinates...")
    
    # Storage
    # edge_polygons: list of dicts with 'polygon', 'row', 'col', 'edges'
    edge_polygons = [] 
    interior_polygons = []
    
    image_lookup = {img['id']: img['file_name'] for img in coco_data['images']}
    
    for i, ann in enumerate(coco_data['annotations']):
        if ann['image_id'] not in image_lookup: continue
        
        fname = image_lookup[ann['image_id']]
        row, col = get_tile_position(fname)
        x_off, y_off = col * TILE_SIZE, row * TILE_SIZE
        
        # Get local polygons
        local_polys = rle_to_polygon(ann.get('segmentation', {}))
        
        for poly in local_polys:
            # Check edges in LOCAL coordinates (0-256)
            bounds = poly.bounds # minx, miny, maxx, maxy
            edges = set()
            if bounds[0] < EDGE_THRESHOLD: edges.add('left')
            if bounds[1] < EDGE_THRESHOLD: edges.add('top')
            if bounds[2] > TILE_SIZE - EDGE_THRESHOLD: edges.add('right')
            if bounds[3] > TILE_SIZE - EDGE_THRESHOLD: edges.add('bottom')
            
            # Transform to GLOBAL coordinates
            from shapely.affinity import translate
            global_poly = translate(poly, xoff=x_off, yoff=y_off)
            
            item = {
                'polygon': global_poly,
                'category_id': ann['category_id'],
                'tile_row': row,
                'tile_col': col,
                'touched_edges': edges
            }
            
            if edges:
                edge_polygons.append(item)
            else:
                interior_polygons.append(item)

    print(f"  Interior (kept as is): {len(interior_polygons)}")
    print(f"  Edge Candidates: {len(edge_polygons)}")

    # Indexing for speed
    # Key: (row, col, 'edge_name') -> indices in edge_polygons list
    edge_index = defaultdict(list)
    for idx, item in enumerate(edge_polygons):
        for edge in item['touched_edges']:
            edge_index[(item['tile_row'], item['tile_col'], edge)].append(idx)

    merged_indices = set()
    final_merged_shapes = []

    print("Step 2: Merging Neighbors...")
    
    for idx, item in enumerate(edge_polygons):
        if idx in merged_indices: continue
        
        # Start a group (flood fill)
        group_indices = {idx}
        stack = [idx]
        
        while stack:
            curr_idx = stack.pop()
            curr = edge_polygons[curr_idx]
            r, c = curr['tile_row'], curr['tile_col']
            
            # Define neighbors to check based on which edges this specific polygon touches
            checks = []
            if 'right' in curr['touched_edges']:  checks.append((r, c+1, 'left'))
            if 'left' in curr['touched_edges']:   checks.append((r, c-1, 'right'))
            if 'bottom' in curr['touched_edges']: checks.append((r+1, c, 'top'))
            if 'top' in curr['touched_edges']:    checks.append((r-1, c, 'bottom'))
            
            for (nr, nc, n_edge) in checks:
                potential_neighbors = edge_index.get((nr, nc, n_edge), [])
                
                curr_geo_buffered = curr['polygon'].buffer(MERGE_BUFFER)
                
                for n_idx in potential_neighbors:
                    if n_idx in group_indices: continue
                    
                    neighbor_geo = edge_polygons[n_idx]['polygon']
                    
                    # Do they touch/overlap with the buffer?
                    if curr_geo_buffered.intersects(neighbor_geo):
                        group_indices.add(n_idx)
                        stack.append(n_idx)
                        # Add the neighbor's edges to the loop (so we can chain A->B->C)
                        # The stack handles this logic automatically

        # Merge the group
        merged_indices.update(group_indices)
        
        polys_to_merge = [edge_polygons[i]['polygon'] for i in group_indices]
        
        if len(polys_to_merge) > 1:
            # Union with buffer to smooth gaps
            union_poly = unary_union([p.buffer(0.5) for p in polys_to_merge]).buffer(-0.5)
        else:
            union_poly = polys_to_merge[0]
            
        final_merged_shapes.append({
            'polygon': make_valid(union_poly),
            'category_id': item['category_id']
        })

    # Combine everything
    all_shapes = interior_polygons + final_merged_shapes
    print(f"Total Buildings after merge: {len(all_shapes)}")

    # Export
    output_annotations = []
    for i, shape in enumerate(all_shapes):
        poly = shape['polygon']
        seg = polygon_to_coco(poly)
        if not seg: continue
        
        x, y, xmax, ymax = poly.bounds
        output_annotations.append({
            "id": i,
            "image_id": 0,
            "category_id": shape['category_id'],
            "segmentation": seg,
            "area": poly.area,
            "bbox": [x, y, xmax-x, ymax-y],
            "iscrowd": 0
        })

    return {
        "images": [{"id": 0, "file_name": "merged_map.png", "width": TILES_PER_ROW*TILE_SIZE, "height": TILES_PER_ROW*TILE_SIZE}],
        "categories": coco_data.get('categories', []),
        "annotations": output_annotations
    }

def main():
    input_json = Path("data/annotated_data/train/_annotations.coco.json")
    output_json = Path("data/annotated_data/merged_buildings.json")
    
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    result = process_coco_annotations(coco_data)
    
    with open(output_json, 'w') as f:
        json.dump(result, f)
    print("Done.")

if __name__ == "__main__":
    main()