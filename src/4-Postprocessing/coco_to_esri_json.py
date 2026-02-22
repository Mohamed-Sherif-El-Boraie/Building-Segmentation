import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import rasterio
from rasterio.transform import xy

# --- CONFIGURATION ---
INPUT_COCO_PATH = Path("data/coco_json_annotations/merged_annotations_downscaled.json")
INPUT_TIF_PATH = Path("data/raw_data/sample.tif")
OUTPUT_PATH = Path("data/esri_json/esri_annotations.json")

def get_tif_metadata(tif_path: Path):
    """
    Opens the TIF to get the Spatial Reference (WKID) and the Affine Transform
    to convert pixel coordinates to map coordinates.
    """
    if not tif_path.exists():
        print(f"Warning: TIF file not found at {tif_path}. Defaulting to pixel coords.")
        return None, None

    with rasterio.open(tif_path) as src:
        # Get Spatial Reference (e.g., 32636)
        crs = src.crs
        wkid = crs.to_epsg() if crs else None
        
        # Get Transform (for converting pixels to meters/degrees)
        transform = src.transform
        
        return wkid, transform

def pixel_poly_to_map_poly(coco_segmentation: List[List[float]], transform) -> List[List[List[float]]]:
    """
    Converts COCO segmentation (pixel coords) to Esri Rings (Map coords)
    using the TIF's transform.
    """
    rings = []
    for seg in coco_segmentation:
        # COCO is [x1, y1, x2, y2, ...]
        # We need to pair them up -> [(x1,y1), (x2,y2)]
        poly_points = []
        for i in range(0, len(seg), 2):
            px, py = seg[i], seg[i+1]
            
            # Convert Pixel (row, col) to Map (x, y)
            # Note: rasterio expects (row, col) which implies (y, x) order for indexing,
            # but transform *usually* takes (x, y) if using affine multiplication.
            # To be safe and strictly follow rasterio logic:
            if transform:
                map_x, map_y = transform * (px, py)
                poly_points.append([map_x, map_y])
            else:
                poly_points.append([px, py])
        
        # Close the ring if needed (Esri requires closed polygons)
        if poly_points and poly_points[0] != poly_points[-1]:
            poly_points.append(poly_points[0])
            
        if len(poly_points) >= 3: # Valid polygon needs at least 3 points
            rings.append(poly_points)
    return rings

def main():
    # 1. Setup paths and load data
    print(f"Loading COCO data from: {INPUT_COCO_PATH}")
    with open(INPUT_COCO_PATH, 'r') as f:
        coco_data = json.load(f)

    # 2. Get Spatial Reference from TIF
    print(f"Reading Spatial Reference from: {INPUT_TIF_PATH}")
    wkid, transform = get_tif_metadata(INPUT_TIF_PATH)
    
    spatial_ref_dict = {
        "wkid": wkid if wkid else 0,
        "latestWkid": wkid if wkid else 0
    }

    # 3. Define the strict Esri JSON Structure (Top Level)
    # This matches your SAM_sample_Det_FeaturesToJSO.JSON exactly
    esri_json = {
        "displayFieldName": "",
        "fieldAliases": {
            "OID": "OID",
            "Class": "Class",
            "Confidence": "Confidence",
            "Shape_Length": "Shape_Length",
            "Shape_Area": "Shape_Area"
        },
        "geometryType": "esriGeometryPolygon",
        "spatialReference": spatial_ref_dict,
        "fields": [
            {"name": "OID", "type": "esriFieldTypeOID", "alias": "OID"},
            {"name": "Class", "type": "esriFieldTypeString", "alias": "Class", "length": 1024},
            {"name": "Confidence", "type": "esriFieldTypeDouble", "alias": "Confidence"},
            {"name": "Shape_Length", "type": "esriFieldTypeDouble", "alias": "Shape_Length"},
            {"name": "Shape_Area", "type": "esriFieldTypeDouble", "alias": "Shape_Area"}
        ],
        "features": []
    }

    # 4. Convert Features
    print("Converting features...")
    annotations = coco_data.get('annotations', [])
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    for idx, ann in enumerate(annotations, 1):
        # Calculate Attributes
        class_name = categories.get(ann.get('category_id'), "Unknown")
        confidence = ann.get('score', 1.0) # Default to 1.0 if no score present
        
        # Geometry Conversion
        segmentation = ann.get('segmentation', [])
        rings = pixel_poly_to_map_poly(segmentation, transform)
        
        if not rings:
            continue

        # Create Feature Object
        feature = {
            "attributes": {
                "OID": idx,
                "Class": class_name,
                "Confidence": confidence,
                "Shape_Length": 0, # ArcGIS usually calculates this automatically
                "Shape_Area": 0    # ArcGIS usually calculates this automatically
            },
            "geometry": {
                "rings": rings
            }
        }
        esri_json["features"].append(feature)

    # 5. Save Output
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    print(f"Saving {len(esri_json['features'])} features to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(esri_json, f, indent=2)

if __name__ == "__main__":
    main()