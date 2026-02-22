import sys
from pathlib import Path

# Add project root to path so we can import from config/ and src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Now that path is set, import from config
from config.config import JSON_PATH

import os
import json
import tempfile
import shutil
import numpy as np
import cv2
import gradio as gr
from datetime import datetime

# ─── Preprocessing helpers (Lightroom-style adjustments) ───────────────────
import rasterio


def adjust_brightness(rgb_uint8, value=0.0):
    """Shift brightness. value in [-1.0, 1.0]. Operates on L channel (LAB)."""
    if abs(value) < 0.001:
        return rgb_uint8
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] += value * 100  # L range is 0-255 in OpenCV LAB
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def adjust_exposure(rgb_uint8, ev=0.0):
    """Multiplicative exposure like camera EV. ev in [-2.0, 2.0]."""
    if abs(ev) < 0.001:
        return rgb_uint8
    factor = 2.0 ** ev
    img = rgb_uint8.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)


def adjust_contrast(rgb_uint8, value=0.0):
    """Contrast around midpoint. value in [-1.0, 1.0]."""
    if abs(value) < 0.001:
        return rgb_uint8
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    mid = np.mean(L)
    L = mid + (L - mid) * (1.0 + value)
    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def adjust_highlights(rgb_uint8, value=0.0):
    """Brighten/darken bright areas (L > 70%). value in [-1.0, 1.0]."""
    if abs(value) < 0.001:
        return rgb_uint8
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    # Smooth mask for highlights (pixels above ~70% brightness)
    highlight_mask = np.clip((L - 170) / 60, 0, 1)  # gradual falloff
    L += value * 50 * highlight_mask
    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def adjust_shadows(rgb_uint8, value=0.0):
    """Lift/lower dark areas (L < 30%). value in [-1.0, 1.0]."""
    if abs(value) < 0.001:
        return rgb_uint8
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    # Smooth mask for shadows (pixels below ~30% brightness)
    shadow_mask = np.clip((80 - L) / 60, 0, 1)  # gradual falloff
    L += value * 50 * shadow_mask
    lab[:, :, 0] = np.clip(L, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def adjust_saturation(rgb_uint8, value=0.0):
    """Color intensity. value in [-1.0, 1.0]."""
    if abs(value) < 0.001:
        return rgb_uint8
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1.0 + value)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def adjust_sharpness(rgb_uint8, value=0.0):
    """Edge crispness via unsharp mask. value in [0.0, 2.0]."""
    if value < 0.01:
        return rgb_uint8
    img = rgb_uint8.astype(np.float32)
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=1.5, sigmaY=1.5)
    detail = img - blurred
    # Suppress noise: only keep details above a small threshold
    noise_gate = np.abs(detail) > 3
    detail = detail * noise_gate
    sharpened = img + value * detail
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(rgb_uint8, clip_limit=2.0, tile_size=8):
    """
    Contrast Limited Adaptive Histogram Equalization.
    clip_limit: Threshold for contrast limiting.
    tile_size: Size of grid for histogram equalization (NxN).
    """
    if clip_limit <= 0:
        return rgb_uint8

    # Convert to LAB to apply CLAHE only on Lightness channel
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def adjust_gamma(rgb_uint8, value=1.0):
    """Non-linear brightness. value in [0.1, 3.0]. 1.0 is neutral."""
    if abs(value - 1.0) < 0.001:
        return rgb_uint8
    invGamma = 1.0 / value
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(rgb_uint8, table)


def apply_denoise(rgb_uint8, strength=0.0):
    """Remove sensor noise. strength in [0.0, 1.0]."""
    if strength < 0.01:
        return rgb_uint8
    # Scale strength to an odd kernel size (3, 5, 7, 9)
    ksize = int(strength * 10)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        return rgb_uint8
    return cv2.medianBlur(rgb_uint8, min(ksize, 9))


def adjust_temp_tint(rgb_uint8, temp=0.0, tint=0.0):
    """
    Adjust White Balance.
    temp: -1.0 (Cool/Blue) to 1.0 (Warm/Yellow)
    tint: -1.0 (Green) to 1.0 (Magenta)
    """
    img = rgb_uint8.astype(np.float32)
    # Temperature: Warm shifts R up, B down. Cool is opposite.
    if abs(temp) > 0.001:
        img[:, :, 0] += temp * 30  # Red
        img[:, :, 2] -= temp * 30  # Blue
    
    # Tint: Magenta shifts R & B up. Green shifts G up.
    if abs(tint) > 0.001:
        img[:, :, 1] -= tint * 20  # Green
        img[:, :, 0] += tint * 10  # Red
        img[:, :, 2] += tint * 10  # Blue
        
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_haze_removal(rgb_uint8, strength=0.0):
    """
    Simplified Haze Removal (Dark Channel Prior logic).
    strength: 0.0 to 1.0
    """
    if strength < 0.01:
        return rgb_uint8
    
    # Estimate haze by finding the minimum across channels
    dark_channel = np.min(rgb_uint8, axis=2)
    haze_intensity = np.mean(dark_channel) * strength
    
    # Subtract haze and re-normalize
    img = rgb_uint8.astype(np.float32)
    img = (img - haze_intensity) / (255.0 - haze_intensity) * 255.0
    
    return np.clip(img, 0, 255).astype(np.uint8)


# ─── Inference helpers (from src/3-Auto-Labeling) ──────────────────────────

def mask_to_polygon(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    return polygons


def mask_to_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def create_coco_annotation(mask, annotation_id, image_id, category_id):
    polygons = mask_to_polygon(mask)
    bbox = mask_to_bbox(mask)
    area = int(np.sum(mask))
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": polygons,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }


def results_to_coco(results, image_path, image_id, categories, annotation_id_start=1):
    img = cv2.imread(str(image_path))
    height, width = img.shape[:2]
    image_info = {
        "id": image_id,
        "file_name": Path(image_path).name,
        "width": width,
        "height": height,
    }
    annotations = []
    annotation_id = annotation_id_start
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            category_id = 1
            if hasattr(results, "boxes") and results.boxes is not None:
                if hasattr(results.boxes, "cls") and len(results.boxes.cls) > i:
                    category_id = int(results.boxes.cls[i].item()) + 1
            annotation = create_coco_annotation(mask, annotation_id, image_id, category_id)
            if annotation["segmentation"]:
                annotations.append(annotation)
                annotation_id += 1
    return image_info, annotations, annotation_id


def draw_overlay(image_rgb, results, color_bgr=(0, 20, 230), alpha=0.6):
    """Draw segmentation overlay on an RGB image, return RGB uint8."""
    overlay = image_rgb.copy()
    if results.masks is not None:
        mask_layer = np.zeros_like(overlay)
        for mask in results.masks.data.cpu().numpy():
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if len(c) >= 3:
                    cv2.fillPoly(mask_layer, [c], color_bgr)
        mask_binary = mask_layer.any(axis=2)
        overlay[mask_binary] = (
            overlay[mask_binary] * (1 - alpha) + mask_layer[mask_binary] * alpha
        ).astype(np.uint8)
    return overlay


# ─── Comparison helpers ────────────────────────────────────────────────────

def polygon_to_mask(segmentation, height, width):
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentation:
        pts = np.array(seg, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


# ─── Global state ──────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.work_dir = tempfile.mkdtemp(prefix="sam3_ui_")
        self.upload_dir = os.path.join(self.work_dir, "uploads")
        self.preprocess_dir = os.path.join(self.work_dir, "preprocessed")
        self.overlay_dir = os.path.join(self.work_dir, "overlays")
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)
        self.predictor = None
        self.coco_json_path = None

    def reset_preprocess(self):
        if os.path.exists(self.preprocess_dir):
            shutil.rmtree(self.preprocess_dir)
        os.makedirs(self.preprocess_dir, exist_ok=True)


state = AppState()


# ─── Core pipeline functions ──────────────────────────────────────────────

def load_image_as_rgb(path):
    """Load TIF (via rasterio) or PNG/JPG (via OpenCV) as RGB uint8."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            # Read first 3 bands
            bands = min(src.count, 3)
            arr = src.read(list(range(1, bands + 1)))  # (C, H, W)
            img = np.transpose(arr, (1, 2, 0))  # (H, W, C)
            if img.dtype != np.uint8:
                # Normalise to uint8
                img = img.astype(np.float32)
                for c in range(img.shape[2]):
                    band = img[:, :, c]
                    lo, hi = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
                    if hi - lo < 1e-6:
                        hi = lo + 1
                    img[:, :, c] = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
                img = img.astype(np.uint8)
            return img
    else:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def handle_upload(files):
    """Save uploaded files and return gallery of originals."""
    if not files:
        return [], "⚠️ No files uploaded."

    # Clear previous uploads
    if os.path.exists(state.upload_dir):
        shutil.rmtree(state.upload_dir)
    os.makedirs(state.upload_dir, exist_ok=True)
    state.reset_preprocess()

    gallery = []
    for f in files:
        src_path = f if isinstance(f, str) else f.name
        dst_path = os.path.join(state.upload_dir, Path(src_path).name)
        shutil.copy2(src_path, dst_path)
        try:
            rgb = load_image_as_rgb(dst_path)
            gallery.append(rgb)
        except Exception as e:
            print(f"Warning: could not preview {dst_path}: {e}")

    return gallery, f"✅ {len(gallery)} image(s) uploaded successfully."


def preprocess_images(brightness, exposure_val, contrast, highlights, shadows, saturation, sharpness, clahe_clip, clahe_tile, gamma, denoise, temp, tint, haze, enable_all):
    """Preprocess all uploaded images with Lightroom-style adjustments."""
    upload_files = sorted(Path(state.upload_dir).glob("*"))
    if not upload_files:
        return [], "⚠️ No images to preprocess. Upload images first."

    state.reset_preprocess()
    gallery = []

    for fpath in upload_files:
        try:
            rgb = load_image_as_rgb(str(fpath))
            
            if not enable_all:
                # Just show original if master disabled
                out = rgb
            else:
                # 1. Atmospheric/Color Correction
                out = apply_haze_removal(rgb, strength=haze)
                out = adjust_temp_tint(out, temp=temp, tint=tint)
                
                # 2. Lighting/Tone
                out = adjust_exposure(out, ev=exposure_val)
                out = adjust_gamma(out, value=gamma)
                out = adjust_brightness(out, value=brightness)
                out = adjust_contrast(out, value=contrast)
                
                # 3. Local Contrast & Details
                if clahe_clip != 0:
                    out = apply_clahe(out, clip_limit=clahe_clip, tile_size=int(clahe_tile))
                
                out = apply_denoise(out, strength=denoise)
                out = adjust_highlights(out, value=highlights)
                out = adjust_shadows(out, value=shadows)
                out = adjust_saturation(out, value=saturation)
                out = adjust_sharpness(out, value=sharpness)

            # Save as PNG for inference compatibility
            out_name = fpath.stem + ".png"
            out_path = os.path.join(state.preprocess_dir, out_name)
            cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            
            # Add pair to gallery for comparison
            gallery.append((rgb, f"Original: {fpath.name}"))
            gallery.append((out, f"Processed: {fpath.name}"))

        except Exception as e:
            print(f"Error preprocessing {fpath.name}: {e}")

    return gallery, f"✅ Preprocessed {len(gallery)} image(s). Adjust params and re-preprocess, or proceed to Inference."


def run_inference(text_prompt, conf_threshold, progress=gr.Progress()):
    """Run SAM3 semantic predictor on all preprocessed images."""
    preprocess_files = sorted(Path(state.preprocess_dir).glob("*.png"))
    if not preprocess_files:
        return [], None, "⚠️ No preprocessed images. Run Preprocess first."

    progress(0, desc="Loading SAM3 model...")

    # Lazy-load predictor
    if state.predictor is None:
        from ultralytics.models.sam import SAM3SemanticPredictor

        model_path = str(PROJECT_ROOT / "models" / "sam3.pt")
        overrides = dict(
            conf=conf_threshold,
            task="segment",
            mode="predict",
            model=model_path,
        )
        state.predictor = SAM3SemanticPredictor(overrides=overrides)
    else:
        state.predictor.args.conf = conf_threshold

    categories = [{"id": 1, "name": "building", "supercategory": "structure"}]
    coco_output = {
        "info": {
            "description": "SAM3 Semantic Segmentation Output",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    image_id = 1
    annotation_id = 1
    overlay_gallery = []

    # Clear overlay dir
    if os.path.exists(state.overlay_dir):
        shutil.rmtree(state.overlay_dir)
    os.makedirs(state.overlay_dir, exist_ok=True)

    for idx, img_path in enumerate(preprocess_files):
        progress((idx + 1) / len(preprocess_files), desc=f"Processing {img_path.name}...")

        state.predictor.set_image(str(img_path))
        results = state.predictor(text=[text_prompt])[0]

        # COCO annotations
        image_info, annotations, annotation_id = results_to_coco(
            results, img_path, image_id, categories, annotation_id
        )
        coco_output["images"].append(image_info)
        coco_output["annotations"].extend(annotations)

        # Overlay image
        rgb = load_image_as_rgb(str(img_path))
        overlay = draw_overlay(rgb, results)
        overlay_gallery.append(overlay)

        # Save overlay
        ov_path = os.path.join(state.overlay_dir, f"{img_path.stem}_overlay.png")
        cv2.imwrite(ov_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        image_id += 1

    # Save COCO JSON
    json_path = os.path.join(state.work_dir, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    state.coco_json_path = json_path

    total_anns = len(coco_output["annotations"])
    msg = f"✅ Inference complete! {len(preprocess_files)} images → {total_anns} building annotations."

    return overlay_gallery, json_path, msg


def compare_results(gt_file, iou_threshold):
    """Compare predicted annotations against ground truth COCO JSON."""
    if state.coco_json_path is None or not os.path.exists(state.coco_json_path):
        return None, "⚠️ No prediction JSON found. Run Inference first."

    msg_prefix = ""
    if gt_file is None:
        # Resolve path relative to project root
        effective_gt_path = PROJECT_ROOT / JSON_PATH
        if not effective_gt_path.exists():
            return None, f"⚠️ Ground truth file not found at {JSON_PATH}. Please upload one manually."
        msg_prefix = f"ℹ️ Using default GT: {JSON_PATH}\n"
        effective_gt_path = str(effective_gt_path)
    else:
        effective_gt_path = gt_file if isinstance(gt_file, str) else gt_file.name

    with open(state.coco_json_path, "r") as f:
        pred_coco = json.load(f)
    with open(effective_gt_path, "r") as f:
        gt_coco = json.load(f)

    # Build lookup: file_name → image_id for both pred and GT
    pred_img_lookup = {img["file_name"]: img for img in pred_coco.get("images", [])}
    gt_img_lookup = {img["file_name"]: img for img in gt_coco.get("images", [])}

    # Group annotations by image_id
    def group_anns(coco_data):
        groups = {}
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            groups.setdefault(img_id, []).append(ann)
        return groups

    pred_anns_by_img = group_anns(pred_coco)
    gt_anns_by_img = group_anns(gt_coco)

    # Build image_id → image_info mappings
    pred_id_to_info = {img["id"]: img for img in pred_coco.get("images", [])}
    gt_id_to_info = {img["id"]: img for img in gt_coco.get("images", [])}

    rows = []
    total_tp, total_fp, total_fn = 0, 0, 0
    total_iou_sum = 0.0
    total_matched = 0

    # Match images by file_name
    all_filenames = set(pred_img_lookup.keys()) | set(gt_img_lookup.keys())

    for fname in sorted(all_filenames):
        pred_img = pred_img_lookup.get(fname)
        gt_img = gt_img_lookup.get(fname)

        if pred_img is None and gt_img is None:
            continue

        width = (pred_img or gt_img)["width"]
        height = (pred_img or gt_img)["height"]

        # Get annotations for this image
        pred_anns = pred_anns_by_img.get(pred_img["id"], []) if pred_img else []
        gt_anns = gt_anns_by_img.get(gt_img["id"], []) if gt_img else []

        # Convert to masks
        pred_masks = []
        for ann in pred_anns:
            if "segmentation" in ann and ann["segmentation"]:
                m = polygon_to_mask(ann["segmentation"], height, width)
                pred_masks.append(m)

        gt_masks = []
        for ann in gt_anns:
            if "segmentation" in ann and ann["segmentation"]:
                m = polygon_to_mask(ann["segmentation"], height, width)
                gt_masks.append(m)

        # Greedy IoU matching
        matched_pred = set()
        matched_gt = set()
        match_ious = []

        if pred_masks and gt_masks:
            iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
            for pi, pm in enumerate(pred_masks):
                for gi, gm in enumerate(gt_masks):
                    iou_matrix[pi, gi] = compute_iou(pm, gm)

            # Greedy matching: pick highest IoU pairs first
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]
                if max_iou < iou_threshold:
                    break
                pi, gi = max_idx
                matched_pred.add(pi)
                matched_gt.add(gi)
                match_ious.append(max_iou)
                iou_matrix[pi, :] = -1
                iou_matrix[:, gi] = -1

        tp = len(matched_pred)
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(match_ious)) if match_ious else 0.0

        rows.append({
            "Image": fname,
            "GT Buildings": len(gt_masks),
            "Pred Buildings": len(pred_masks),
            "Correct (TP)": tp,
            "False Pos (FP)": fp,
            "Missed (FN)": fn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
            "Mean IoU": round(mean_iou, 3),
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_iou_sum += sum(match_ious)
        total_matched += len(match_ious)

    # Overall row
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )
    overall_iou = total_iou_sum / total_matched if total_matched > 0 else 0.0

    rows.append({
        "Image": "── OVERALL ──",
        "GT Buildings": sum(r["GT Buildings"] for r in rows),
        "Pred Buildings": sum(r["Pred Buildings"] for r in rows),
        "Correct (TP)": total_tp,
        "False Pos (FP)": total_fp,
        "Missed (FN)": total_fn,
        "Precision": round(overall_prec, 3),
        "Recall": round(overall_rec, 3),
        "F1": round(overall_f1, 3),
        "Mean IoU": round(overall_iou, 3),
    })

    import pandas as pd

    df = pd.DataFrame(rows)
    msg = msg_prefix + (
        f"✅ Comparison complete! Matched {len(all_filenames)} image(s).\n"
        f"Overall — Precision: {overall_prec:.3f} | Recall: {overall_rec:.3f} | "
        f"F1: {overall_f1:.3f} | Mean IoU: {overall_iou:.3f}"
    )
    return df, msg


# ══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
)

CSS = """
    .status-box { font-size: 14px !important; }
    .gr-button-primary { min-height: 48px !important; font-weight: 600 !important; }
"""

with gr.Blocks(title="SAM3 Building Segmentation") as demo:

    gr.Markdown(
        """
        # 🏗️ SAM3 Building Segmentation Pipeline
        Upload satellite images → Preprocess → Run SAM3 Inference → Compare with Ground Truth
        """
    )

    # ── Tab 1: Upload & Preprocess ─────────────────────────────────────
    with gr.Tab("1️⃣  Upload & Preprocess"):
        with gr.Row():
            with gr.Column(scale=1):
                upload_input = gr.File(
                    label="Upload Images (TIF / PNG)",
                    file_count="multiple",
                    file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                upload_btn = gr.Button("📤 Upload", variant="primary")
                upload_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")

            with gr.Column(scale=2):
                upload_gallery = gr.Gallery(
                    label="Uploaded Images",
                    columns=4,
                    height=350,
                    object_fit="contain",
                )

        gr.Markdown("### ⚙️ Image Adjustments")

        with gr.Row():
            enable_preprocess = gr.Checkbox(label="✅ Enable Preprocessing", value=True, scale=1)
            reset_btn = gr.Button("🔄 Reset All to Defaults", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column():
                brightness_slider = gr.Slider(-5.0, 5.0, value=0.05, step=0.01, label="☀️ Brightness")
                exposure_slider = gr.Slider(-5.0, 5.0, value=0.35, step=0.05, label="📷 Exposure")
                contrast_slider = gr.Slider(-5.0, 5.0, value=0.10, step=0.01, label="🔲 Contrast")
                highlights_slider = gr.Slider(-5.0, 5.0, value=0.10, step=0.01, label="🔆 Highlights")
            with gr.Column():
                shadows_slider = gr.Slider(-5.0, 5.0, value=-0.10, step=0.01, label="🌑 Shadows")
                saturation_slider = gr.Slider(-5.0, 5.0, value=0.10, step=0.01, label="🎨 Saturation")
                sharpness_slider = gr.Slider(-5.0, 5.0, value=0.05, step=0.01, label="🔪 Sharpness")
            with gr.Column():
                clahe_clip_slider = gr.Slider(0.0, 10.0, value=2.0, step=0.1, label="⚡ CLAHE Clip Limit (Local Contrast)")
                clahe_tile_slider = gr.Slider(1, 32, value=8, step=1, label="▦ CLAHE Tile Size")
            with gr.Column():
                gamma_slider = gr.Slider(-3.0, 3.0, value=1.0, step=0.05, label="🌈 Gamma (Non-linear)")
                denoise_slider = gr.Slider(-5.0, 5.0, value=0.0, step=0.05, label="🧹 Denoise (Smooth)")
            with gr.Column():
                temp_slider = gr.Slider(-5.0, 5.0, value=0.0, step=0.05, label="🌡️ Temperature (Blue-Yellow)")
                tint_slider = gr.Slider(-5.0, 5.0, value=0.0, step=0.05, label="🧪 Tint (Green-Magenta)")
                haze_slider = gr.Slider(-5.0, 5.0, value=0.0, step=0.05, label="🌫️ Haze Reduction")

        with gr.Row():
            preprocess_btn = gr.Button("🔄 Preprocess", variant="primary", scale=2)
            preprocess_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", scale=3)

        preprocess_gallery = gr.Gallery(
            label="Comparison: Original (Left) vs Processed (Right)",
            columns=2,
            height=800,
            object_fit="contain",
        )

    # ── Tab 2: Inference ───────────────────────────────────────────────
    with gr.Tab("2️⃣  Inference"):
        with gr.Row():
            text_prompt = gr.Textbox(
                label="Text Prompt",
                value="Square Buildings or Rectangular Buildings or Odd-shaped Buildings",
                scale=3,
            )
            conf_slider = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="Confidence Threshold", scale=1)

        with gr.Row():
            inference_btn = gr.Button("🚀 Run Inference", variant="primary", scale=2)
            inference_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", scale=3)

        inference_gallery = gr.Gallery(
            label="Segmentation Results (Overlay)",
            columns=4,
            height=450,
            object_fit="contain",
        )

        json_download = gr.File(label="📥 Download Annotations JSON", interactive=False)

    # ── Tab 3: Compare ─────────────────────────────────────────────────
    with gr.Tab("3️⃣  Compare"):
        with gr.Row():
            gt_upload = gr.File(
                label=f"Upload Ground Truth (Optional - Default: {JSON_PATH})",
                file_types=[".json"],
                type="filepath",
            )
            iou_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="IoU Threshold")

        with gr.Row():
            compare_btn = gr.Button("📊 Compare", variant="primary", scale=2)
            compare_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box", scale=3)

        compare_table = gr.Dataframe(
            label="Comparison Metrics",
            headers=[
                "Image", "GT Buildings", "Pred Buildings",
                "Correct (TP)", "False Pos (FP)", "Missed (FN)",
                "Precision", "Recall", "F1", "Mean IoU",
            ],
            interactive=False,
            wrap=True,
        )

    # ── Event wiring ───────────────────────────────────────────────────
    upload_btn.click(
        fn=handle_upload,
        inputs=[upload_input],
        outputs=[upload_gallery, upload_status],
    )

    # Reset button events
    reset_btn.click(
        fn=lambda: [0.05, 0.35, 0.10, 0.10, -0.10, 0.10, 0.05, 2.0, 8, 1.0, 0.0, 0.0, 0.0, 0.0, True],
        inputs=[],
        outputs=[brightness_slider, exposure_slider, contrast_slider, highlights_slider, shadows_slider, saturation_slider, sharpness_slider, clahe_clip_slider, clahe_tile_slider, gamma_slider, denoise_slider, temp_slider, tint_slider, haze_slider, enable_preprocess],
    )

    preprocess_btn.click(
        fn=preprocess_images,
        inputs=[brightness_slider, exposure_slider, contrast_slider, highlights_slider, shadows_slider, saturation_slider, sharpness_slider, clahe_clip_slider, clahe_tile_slider, gamma_slider, denoise_slider, temp_slider, tint_slider, haze_slider, enable_preprocess],
        outputs=[preprocess_gallery, preprocess_status],
    )

    inference_btn.click(
        fn=run_inference,
        inputs=[text_prompt, conf_slider],
        outputs=[inference_gallery, json_download, inference_status],
    )

    compare_btn.click(
        fn=compare_results,
        inputs=[gt_upload, iou_threshold],
        outputs=[compare_table, compare_status],
    )


# ── Launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=THEME,
        css=CSS,
    )
