import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import json
import numpy as np
from pycocotools.coco import COCO
from shapely.geometry import Polygon
import networkx as nx

# --- Configuration ---
INPUT_JSON = 'data/annotated_data/train/_annotations.coco.json'
OUTPUT_JSON = 'data/annotated_data/train/_annotations_with_ids.coco.json'
TILE_SIZE = 256
GRID_WIDTH = 14  # Adjust this to match your actual grid dimensions

def get_grid_coords(file_name):
    """Extracts row and col from filename like 'tile_0021.png'."""
    try:
        tile_idx = int(''.join(filter(str.isdigit, file_name)))
        row = tile_idx // GRID_WIDTH
        col = tile_idx % GRID_WIDTH
        return row, col
    except ValueError:
        return 0, 0

def main():
    # 1. Load the original JSON data
    print(f"Loading {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        coco_data = json.load(f)
    
    coco = COCO(INPUT_JSON)
    all_anns = coco.loadAnns(coco.getAnnIds())
    
    G = nx.Graph()
    global_polygons = {}

    # 2. Map local coordinates to global space
    print("Mapping fragments to global coordinates...")
    for ann in all_anns:
        img_info = coco.loadImgs(ann['image_id'])[0]
        row, col = get_grid_coords(img_info['file_name'])
        
        offset_x = col * TILE_SIZE
        offset_y = row * TILE_SIZE
        
        if 'segmentation' in ann and isinstance(ann['segmentation'], list):
            # Process each polygon segment for the annotation
            for seg in ann['segmentation']:
                if len(seg) < 6: continue # Skip invalid polygons
                
                pts = np.array(seg).reshape(-1, 2)
                pts[:, 0] += offset_x
                pts[:, 1] += offset_y
                
                # Buffer by 1.5 pixels to ensure overlap at tile boundaries
                poly = Polygon(pts).buffer(1.5)
                
                if poly.is_valid:
                    global_polygons[ann['id']] = poly
                    G.add_node(ann['id'])

    # 3. Find intersections (Spatial Join)
    print("Analyzing spatial relationships (this may take a moment)...")
    ann_ids = list(global_polygons.keys())
    for i in range(len(ann_ids)):
        for j in range(i + 1, len(ann_ids)):
            id_a, id_b = ann_ids[i], ann_ids[j]
            # Check if global footprints intersect
            if global_polygons[id_a].intersects(global_polygons[id_b]):
                G.add_edge(id_a, id_b)

    # 4. Group fragments and assign Building IDs
    connected_buildings = list(nx.connected_components(G))
    # Map: {original_ann_id: unique_building_id}
    id_map = {}
    for b_id, group in enumerate(connected_buildings):
        for ann_id in group:
            id_map[ann_id] = b_id

    # 5. Update the JSON structure
    print("Updating JSON metadata...")
    for ann in coco_data['annotations']:
        # Assign the building_id, default to a new ID if not in a group
        ann['building_id'] = id_map.get(ann['id'], ann['id'] + 100000)

    # 6. Save the results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"\nSuccess!")
    print(f"Total Unique Buildings Found: {len(connected_buildings)}")
    print(f"Saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()