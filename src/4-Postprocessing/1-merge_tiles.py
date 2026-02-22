import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import rasterio
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
from config.config import *
from config.logger import get_logger

logger = get_logger("merge_tiles")

# --- CONFIG ---
TILES_JSON_DIR = SEGMENTATION_OUTPUT_DIR
TILES_TIF_DIR = OUTPUT_TIF_TILES_DIR
OUTPUT_JSON = "data/annotated_data/merged_buildings.json"
IOU_THRESHOLD = 0.3      # true duplicates (same shape)
IOMIN_THRESHOLD = 0.4     # partial overlaps (contained inside larger)


def get_tile_pixel_offset(tile_tif_path, reference_tif_path):
    """Get exact pixel offset using GeoTIFF transforms.
    
    This is more reliable than filename parsing — it uses the actual
    geo-coordinates embedded in each tile.
    """
    with rasterio.open(reference_tif_path) as ref_src:
        ref_transform = ref_src.transform

    with rasterio.open(tile_tif_path) as tile_src:
        tile_transform = tile_src.transform

    # Tile's top-left corner in world coordinates
    tile_x_world = tile_transform.c
    tile_y_world = tile_transform.f

    # Convert to pixel coordinates in the reference image
    inv_ref = ~ref_transform
    x_offset, y_offset = inv_ref * (tile_x_world, tile_y_world)

    return int(round(x_offset)), int(round(y_offset))


def shift_annotation(ann, x_offset, y_offset):
    """Shift segmentation and bbox from tile-local to global coordinates."""
    new_ann = ann.copy()
    new_segs = []
    for seg in ann["segmentation"]:
        coords = np.array(seg, dtype=np.float64)
        coords[0::2] += x_offset
        coords[1::2] += y_offset
        new_segs.append(coords.tolist())
    new_ann["segmentation"] = new_segs

    if "bbox" in ann:
        bbox = ann["bbox"].copy()
        bbox[0] += x_offset
        bbox[1] += y_offset
        new_ann["bbox"] = bbox

    return new_ann


def coco_seg_to_polygon(seg):
    """Convert COCO segmentation [x1,y1,x2,y2,...] to Shapely Polygon."""
    coords = np.array(seg).reshape(-1, 2)
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area == 0:
        return None
    return poly


def compute_iou(poly1, poly2):
    """Compute IoU between two Shapely polygons."""
    if poly1 is None or poly2 is None:
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def nms_deduplicate(annotations, iou_threshold=IOU_THRESHOLD, iomin_threshold=IOMIN_THRESHOLD):
    """Remove duplicate detections using IoU + IoMin (intersection over minimum area).
    
    IoU catches true duplicates (same shape, same building).
    IoMin catches partial overlaps (smaller detection inside larger one).
    """
    polys = []
    for ann in annotations:
        poly = coco_seg_to_polygon(ann["segmentation"][0]) if ann["segmentation"] else None
        polys.append(poly)

    n = len(annotations)
    suppressed = [False] * n

    # Sort by area descending — keep larger/more complete detections
    indices = sorted(range(n), key=lambda i: annotations[i].get("area", 0), reverse=True)

    # Build spatial index
    valid_polys = [(i, polys[i]) for i in range(n) if polys[i] is not None]
    if len(valid_polys) == 0:
        return []

    poly_list = [p for _, p in valid_polys]
    idx_list = [i for i, _ in valid_polys]
    tree = STRtree(poly_list)

    tree_idx_to_ann_idx = {k: idx_list[k] for k in range(len(idx_list))}
    ann_idx_to_tree_idx = {v: k for k, v in tree_idx_to_ann_idx.items()}

    for idx_i in indices:
        if suppressed[idx_i] or polys[idx_i] is None:
            continue
        if idx_i not in ann_idx_to_tree_idx:
            continue

        candidates = tree.query(polys[idx_i])
        for tree_j in candidates:
            idx_j = tree_idx_to_ann_idx[tree_j]
            if idx_j == idx_i or suppressed[idx_j]:
                continue

            try:
                inter_area = polys[idx_i].intersection(polys[idx_j]).area
            except Exception:
                continue

            area_i = polys[idx_i].area
            area_j = polys[idx_j].area
            union_area = area_i + area_j - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            iomin = inter_area / min(area_i, area_j) if min(area_i, area_j) > 0 else 0

            # Suppress if either condition met:
            # 1) IoU > threshold (true duplicate)
            # 2) IoMin > threshold (smaller is mostly inside larger)
            if iou > iou_threshold or iomin > iomin_threshold:
                # Suppress the smaller one (idx_j is always smaller since sorted by area desc)
                if area_i >= area_j:
                    suppressed[idx_j] = True
                else:
                    suppressed[idx_i] = True
                    break

    merged = [ann for i, ann in enumerate(annotations) if not suppressed[i] and polys[i] is not None]
    logger.info(f"NMS+IoMin: {n} → {len(merged)} (removed {n - len(merged)} duplicates)")
    return merged

def main():
    json_dir = Path(TILES_JSON_DIR)
    tif_dir = Path(TILES_TIF_DIR)
    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {json_dir}")
        return

    # Find reference TIF (the full enhanced image) for coordinate mapping
    reference_tif = ENHANCED_IMAGE
    if not Path(reference_tif).exists():
        logger.error(f"Reference TIF not found: {reference_tif}")
        return

    # Build tile filename → pixel offset lookup using geo-transforms
    logger.info("Computing tile offsets from GeoTIFF metadata...")
    tile_offsets = {}
    for tif_path in sorted(tif_dir.glob("*.tif")):
        try:
            x_off, y_off = get_tile_pixel_offset(str(tif_path), reference_tif)
            tile_offsets[tif_path.name] = (x_off, y_off)
        except Exception as e:
            logger.warning(f"Could not read offset for {tif_path.name}: {e}")

    logger.info(f"Computed offsets for {len(tile_offsets)} tiles")

    all_annotations = []
    categories = None
    global_ann_id = 1

    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)

        if categories is None and "categories" in data:
            categories = data["categories"]

        # Build image_id → filename lookup
        image_id_to_filename = {}
        for img in data.get("images", []):
            image_id_to_filename[img["id"]] = img["file_name"]

        tile_count = 0
        for ann in data.get("annotations", []):
            filename = image_id_to_filename.get(ann["image_id"], "")

            # Get offset from geo-transform (fallback to 0,0 if not found)
            x_off, y_off = tile_offsets.get(filename, (0, 0))
            if (x_off, y_off) == (0, 0) and filename:
                logger.warning(f"No geo-offset for {filename}, annotations may be misaligned")

            shifted = shift_annotation(ann, x_off, y_off)
            shifted["id"] = global_ann_id
            shifted["source_tile"] = filename
            all_annotations.append(shifted)
            global_ann_id += 1
            tile_count += 1

        logger.info(f"Loaded {tile_count} annotations from {json_path.name}")

    logger.info(f"Total annotations before dedup: {len(all_annotations)}")

    # Deduplicate overlapping annotations (NMS only — no union merge)
    merged_annotations = nms_deduplicate(all_annotations)

    # Reassign IDs
    for i, ann in enumerate(merged_annotations, 1):
        ann["id"] = i
        ann.pop("source_tile", None)

    # Build output COCO
    output = {
        "info": {"description": "Merged tile annotations"},
        "licenses": [],
        "images": [],
        "annotations": merged_annotations,
        "categories": categories or [{"id": 1, "name": "building"}],
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved {len(merged_annotations)} merged annotations to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()