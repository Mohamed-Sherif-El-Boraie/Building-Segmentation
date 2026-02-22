import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# --- CONFIGURATION ---
INPUT_JSON = "data/output/processed_buildings.json"
OUTPUT_JSON = "data/output/unions_buildings.json"
MIN_AREA_PX = 111  # e.g., 10m^2 at 0.3m resolution
SIMPLIFY_TOL = 0.5

def merge_overlapping_annotations(data):
    # Map annotations to their respective images
    image_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    new_annotations = []
    ann_id_counter = 1

    for img_id, anns in image_to_anns.items():
        polygons = []
        for ann in anns:
            # Handle multiple segmentation paths in one annotation
            for seg in ann['segmentation']:
                poly = Polygon(np.array(seg).reshape(-1, 2))
                if not poly.is_valid: poly = make_valid(poly)
                polygons.append(poly)

        # 1. UNION: This merges all overlapping polygons into one MultiPolygon/Polygon
        merged_geometry = unary_union(polygons)

        # 2. EXPLODE: Separate non-touching buildings back into individual items
        if isinstance(merged_geometry, Polygon):
            final_geoms = [merged_geometry]
        else:
            final_geoms = list(merged_geometry.geoms)

        for geom in final_geoms:
            # 3. POST-PROCESSING (Your requested steps)
            # Opening/Closing & Smoothing via small buffers
            geom = geom.buffer(0.5).buffer(-0.5) 
            
            # Ramer-Douglas-Peucker simplification
            geom = geom.simplify(SIMPLIFY_TOL, preserve_topology=True)

            # 4. FILTER: Remove small objects
            if geom.area < MIN_AREA_PX:
                continue

            # Convert back to COCO format
            if geom.geom_type == 'Polygon' and not geom.is_empty:
                segmentation = [list(np.array(geom.exterior.coords).ravel())]
                new_annotations.append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": anns[0]['category_id'],
                    "segmentation": segmentation,
                    "area": geom.area,
                    "bbox": list(geom.bounds), # [minx, miny, maxx, maxy]
                    "iscrowd": 0
                })
                ann_id_counter += 1

    data['annotations'] = new_annotations
    return data

# Execute
with open(INPUT_JSON, 'r') as f:
    coco_data = json.load(f)

processed_data = merge_overlapping_annotations(coco_data)

with open(OUTPUT_JSON, 'w') as f:
    json.dump(processed_data, f, indent=4)
