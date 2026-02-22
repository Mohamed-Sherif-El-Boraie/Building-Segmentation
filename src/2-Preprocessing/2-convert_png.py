import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import os
from PIL import Image
from config.config import OUTPUT_TIF_TILES_DIR, OUTPUT_PNG_TILES_DIR
from config.logger import get_logger

logger = get_logger("convert_png")


def convert_tif_to_png(input_path, output_path):
    """Convert a single TIF image to RGB PNG."""
    with Image.open(input_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(output_path, 'PNG')


def batch_convert_tif_to_png(tif_dir, png_dir):
    """Convert all TIF files in tif_dir to PNG files in png_dir."""
    os.makedirs(png_dir, exist_ok=True)

    tif_files = sorted([f for f in os.listdir(tif_dir) if f.lower().endswith('.tif')])

    if not tif_files:
        logger.warning(f"No TIF files found in: {tif_dir}")
        return

    logger.info(f"Converting {len(tif_files)} TIF tiles to PNG...")

    for tif_file in tif_files:
        tif_path = os.path.join(tif_dir, tif_file)
        png_file = tif_file.rsplit('.', 1)[0] + '.png'
        png_path = os.path.join(png_dir, png_file)

        try:
            convert_tif_to_png(tif_path, png_path)
        except Exception as e:
            logger.error(f"Error converting {tif_file}: {e}")

    logger.info(f"Conversion complete! Saved {len(tif_files)} PNG tiles to: {png_dir}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_PNG_TILES_DIR, exist_ok=True)
    batch_convert_tif_to_png(OUTPUT_TIF_TILES_DIR, OUTPUT_PNG_TILES_DIR)
