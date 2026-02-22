import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2
import whitebox
from skimage import color, exposure


from config.config import *
from config.logger import get_logger

logger = get_logger("satellite_preprocess_tif")
wbt = whitebox.WhiteboxTools()

def percentile_stretch(img, mask=None, low=2, high=98):
    """
    img: (H, W, C) float or uint8
    mask: (H, W) True = valid pixels
    returns float32 image in [0,1]
    """
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    for c in range(img.shape[2]):
        band = img[:, :, c]

        if mask is not None:
            vals = band[mask]
        else:
            vals = band.reshape(-1)

        if vals.size == 0:
            out[:, :, c] = 0
            continue

        lo = np.percentile(vals, low)
        hi = np.percentile(vals, high)

        if hi - lo < 1e-6:
            out[:, :, c] = 0
        else:
            stretched = (band - lo) / (hi - lo)
            out[:, :, c] = np.clip(stretched, 0, 1)

    return out


def apply_clahe_lab(rgb_uint8, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to L channel only (LAB space). Input must be RGB uint8.
    """
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L2 = clahe.apply(L)

    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def unsharp_mask(rgb_uint8, amount=0.4, radius=1.0, threshold=3):
    """
    Mild sharpening using unsharp mask.
    amount: strength (0.2-0.8 typical)
    radius: blur radius in pixels (1-2 typical)
    threshold: ignore tiny differences to avoid sharpening noise (0-10)
    """
    img = rgb_uint8.astype(np.float32)

    # Gaussian blur (radius -> sigma)
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=radius, sigmaY=radius)

    # "detail" layer
    detail = img - blurred

    if threshold and threshold > 0:
        # only keep details bigger than threshold
        mask = np.abs(detail) > threshold
        detail = detail * mask

    sharpened = img + amount * detail
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_wbt_topographic_correction(in_tif, out_tif, dem_path):
    """
    Removes terrain shadows using C-Correction.
    Requires a DEM file covering the same area.
    """
    logger.info("Starting WhiteboxTools C-Correction...")
    wbt.topographic_correction(
        dem=dem_path,
        imagery=in_tif,
        output=out_tif,
        method="C-Correction",
        azimuth=142.5, # Adjust based on your satellite metadata
        altitude=52.2  # Adjust based on your satellite metadata
    )
    return out_tif

def apply_skimage_shadow_lift(rgb_uint8):
    """
    Lifts shadows specifically in CIELCh space.
    """
    # Convert to LAB then to LCh (Luminance, Chroma, Hue)
    lab = color.rgb2lab(rgb_uint8)
    lch = color.lab2lch(lab)
    
    # L channel is lch[:,:,0]. We rescale only the dark intensity.
    # This lifts the 'black' floor of the image specifically.
    l_channel = lch[:, :, 0]
    
    # Rescale intensity: maps low values (shadows) to slightly higher values
    # without affecting the highlights as much as global gamma.
    l_lifted = exposure.rescale_intensity(l_channel, in_range=(0, 85), out_range=(10, 90))
    
    lch[:, :, 0] = l_lifted
    
    # Convert back to RGB
    lab_new = color.lch2lab(lch)
    rgb_new = color.lab2rgb(lab_new)
    
    return (rgb_new * 255).astype(np.uint8)


# def enhance_geotiff(
#     in_tif: str,
#     out_tif: str,
#     rgb_order=(1, 2, 3),
#     low=5,
#     high=95,
#     gamma=1.8,
#     use_clahe=True,
#     clahe_clip=4,
#     clahe_tile=(16, 16),
#     denoise_ksize=1,
#     denoise_first=False,
#     use_sharpen=True,
#     sharpen_amount=0.6,
#     sharpen_radius=2.0,
#     sharpen_threshold=5,
#     compress="lzw",
# ):
#     with rasterio.open(in_tif) as src:
#         profile = src.profile.copy()
#         nodata = src.nodata

#         # Read only RGB bands (avoid NIR/alpha confusion)
#         arr = src.read(list(rgb_order))  # (3, H, W)

#     # Convert to (H, W, 3)
#     img = np.transpose(arr, (1, 2, 0))

#     # Mask nodata (common for satellite borders)
#     if nodata is not None:
#         # valid if ANY channel is not nodata (less strict, safer for real zeros in a channel)
#         valid_mask = np.any(img != nodata, axis=2)
#     else:
#         valid_mask = np.ones(img.shape[:2], dtype=bool)

#     # 1) Percentile stretch -> float [0,1]
#     stretched = percentile_stretch(img, mask=valid_mask, low=low, high=high)

#     # 2) Gamma correction (optional)
#     if gamma != 1.0:
#         stretched = np.clip(stretched ** (1.0 / gamma), 0, 1)

#     # Convert to uint8 for OpenCV ops / saving
#     out_rgb = (stretched * 255.0).round().astype(np.uint8)

#     # Optional denoise BEFORE CLAHE (useful if image is noisy)
#     if denoise_first and denoise_ksize and denoise_ksize > 1:
#         if denoise_ksize % 2 == 0:
#             logger.error("denoise_ksize must be odd (3,5,7,...)")
#         out_rgb = cv2.medianBlur(out_rgb, denoise_ksize)

#     # 3) CLAHE on brightness only (LAB)
#     if use_clahe:
#         out_rgb = apply_clahe_lab(
#             out_rgb,
#             clip_limit=clahe_clip,
#             tile_grid_size=clahe_tile
#         )

#     # Optional denoise AFTER CLAHE (reduces grain CLAHE can add)
#     if (not denoise_first) and denoise_ksize and denoise_ksize > 1:
#         if denoise_ksize % 2 == 0:
#             logger.error("denoise_ksize must be odd (3,5,7,...)")
#         out_rgb = cv2.medianBlur(out_rgb, denoise_ksize)


#     # Mild sharpening (recommended after denoise/CLAHE)
#     if use_sharpen:
#         out_rgb = unsharp_mask(
#             out_rgb,
#             amount=sharpen_amount,
#             radius=sharpen_radius,
#             threshold=sharpen_threshold
#         )

#     # Keep nodata pixels as 0
#     out_rgb[~valid_mask] = 0

#     # Back to (3, H, W)
#     out_arr = np.transpose(out_rgb, (2, 0, 1))

#     # Write GeoTIFF with same spatial metadata
#     profile.update(dtype="uint8", count=3, compress=compress, photometric="RGB", nodata=0)

#     os.makedirs(os.path.dirname(out_tif), exist_ok=True)
#     with rasterio.open(out_tif, "w", **profile) as dst:
#         dst.write(out_arr)

#     logger.info(f"Saved enhanced GeoTIFF: {out_tif}")

def enhance_geotiff(in_tif, out_tif, dem_path=None, rgb_order=(1, 2, 3)):
    # 1. TOPOGRAPHIC CORRECTION (Optional - Requires DEM)
    current_input = in_tif
    if dem_path and os.path.exists(dem_path):
        topo_out = in_tif.replace(".tif", "_topo_corrected.tif")
        wbt.topographic_correction(dem=dem_path, imagery=in_tif, output=topo_out, method="C-Correction")
        current_input = topo_out

    # 2. WINDOWED PROCESSING (For 20K x 15K Pixels)
    with rasterio.open(current_input) as src:
        profile = src.profile.copy()
        profile.update(dtype="uint8", count=3, compress="lzw", nodata=0)

        with rasterio.open(out_tif, "w", **profile) as dst:
            # Process in blocks (typically 256x256 or 512x512)
            for _, window in src.block_windows():
                # Read 3 bands for this window
                img_block = src.read(list(rgb_order), window=window)
                img_block = np.transpose(img_block, (1, 2, 0)) # To (H, W, 3)

                # Skip processing if block is empty/nodata
                if np.all(img_block == (src.nodata or 0)):
                    dst.write(np.transpose(img_block, (2, 0, 1)), window=window)
                    continue

                # --- ADVANCED ENHANCEMENT ---
                # A) Scikit-Image Shadow Lift
                enhanced = apply_skimage_shadow_lift(img_block)
                
                # B) CLAHE (Local Contrast)
                lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
                l = clahe.apply(l)
                enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

                # Write window back to disk
                dst.write(np.transpose(enhanced, (2, 0, 1)), window=window)

    print(f"Enhancement complete: {out_tif}")

def tile_geotiff(input_path, output_dir, tile_size=512, overlap=128):
    os.makedirs(output_dir, exist_ok=True)
    stride = tile_size - overlap
    
    with rasterio.open(input_path) as src:
        img_width = src.width
        img_height = src.height
        meta = src.meta.copy()

        tile_count = 0
        skipped_count = 0

        for row, y in enumerate(range(0, img_height, stride)):
            for col, x in enumerate(range(0, img_width, stride)):
                
                window = Window(x, y, tile_size, tile_size)
                
                if x + tile_size <= img_width and y + tile_size <= img_height:
                    tile_data = src.read(window=window)
                    tile_transform = src.window_transform(window)

                    # Check if tile is empty (all zeros)
                    if np.all(tile_data == 0):
                        skipped_count += 1
                        continue
                    
                    meta.update({
                        "driver": "GTiff",
                        "height": tile_size,
                        "width": tile_size,
                        "transform": tile_transform,
                        "compress": "lzw" 
                    })
                    
                    tile_filename = f"tile_{row:04d}_{col:04d}.tif"
                    tile_path = os.path.join(output_dir, tile_filename)
                    
                    with rasterio.open(tile_path, "w", **meta) as dest:
                        dest.write(tile_data)
                    
                    tile_count += 1

        logger.info(f"Tiling complete! Created {tile_count} GeoTIFF tiles at: {output_dir}")


if __name__ == "__main__":
    
    raw_image_path = INPUT_TIF
    enhanced_image_path = ENHANCED_IMAGE
    tiles_output_directory = OUTPUT_TIF_TILES_DIR

    # Enhance the Image 
    # enhance_geotiff(
    #     in_tif=raw_image_path,
    #     out_tif=enhanced_image_path,
    # )
    
    # Tile the Enhanced Image
    tile_geotiff(
        input_path=raw_image_path, 
        output_dir=tiles_output_directory,
        tile_size=TILE_SIZE,
        overlap=OVERLAP
    )
    
