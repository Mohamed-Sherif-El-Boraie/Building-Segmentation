import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

# --- CONFIGURATION ---
INPUT_JSON = "data/annotated_data/merged_buildings.json"
OUTPUT_JSON = "data/output/processed_buildings.json"
MIN_AREA_M2 = 10.0  # Threshold in square meters
PIXEL_SIZE = 0.30630739265694024    # Adjust to your TIF resolution
BUFFER_DIST = 1   # Buffer distance for Opening/Closing in pixels
SIMPLIFY_TOL = 3  # Ramer-Douglas-Peucker tolerance

def process_annotations(data):
    new_annotations = []
    area_threshold_px = MIN_AREA_M2 / (PIXEL_SIZE ** 2)

    for ann in data['annotations']:
        # 1. Convert COCO list [x1, y1, x2, y2...] to Shapely Polygon
        coords = np.array(ann['segmentation'][0]).reshape(-1, 2)
        if len(coords) < 3: continue
        poly = Polygon(coords)
        if not poly.is_valid: poly = make_valid(poly)

        # STEP 1: Opening (Erosion -> Dilation) to remove small protrusions
        # STEP 2: Closing (Dilation -> Erosion) to fill small holes/gaps
        # In Shapely, this is done via buffering
        poly = poly.buffer(-BUFFER_DIST).buffer(BUFFER_DIST) # Opening
        poly = poly.buffer(BUFFER_DIST).buffer(-BUFFER_DIST) # Closing

        # STEP 3: Refinement (Ramer-Douglas-Peucker)
        # Simplifies jagged edges by removing redundant vertices
        poly = poly.simplify(SIMPLIFY_TOL, preserve_topology=True)

        # STEP 4: Edge Smoothing (Optional: Polynomial smoothing)
        # Using a small positive buffer followed by a negative buffer with 
        # higher resolution 'join_style' can smooth sharp vertices.
        poly = poly.buffer(0.5, join_style=2).buffer(-0.5, join_style=2)

        # STEP 5: Remove small objects
        if poly.area < area_threshold_px:
            continue

        # Convert back to COCO format
        if poly.geom_type == 'Polygon':
            out_coords = list(np.array(poly.exterior.coords).ravel())
            ann['segmentation'] = [out_coords]
            ann['area'] = poly.area
            ann['bbox'] = list(poly.bounds) # [minx, miny, maxx, maxy]
            new_annotations.append(ann)
        elif poly.geom_type == 'MultiPolygon':
            # If a building split, take the largest part or keep all as separate
            for p in poly.geoms:
                if p.area >= area_threshold_px:
                    new_ann = ann.copy()
                    new_ann['segmentation'] = [list(np.array(p.exterior.coords).ravel())]
                    new_ann['area'] = p.area
                    new_annotations.append(new_ann)

    data['annotations'] = new_annotations
    return data

# Load, Process, and Save
with open(INPUT_JSON, 'r') as f:
    coco_data = json.load(f)

processed_data = process_annotations(coco_data)

with open(OUTPUT_JSON, 'w') as f:
    json.dump(processed_data, f)