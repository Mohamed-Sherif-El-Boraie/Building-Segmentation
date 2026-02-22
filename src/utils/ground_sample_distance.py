import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import rasterio

# Use the path to one of your *original* TIF files
filepath = 'data/raw_data/sample.tif'

with rasterio.open(filepath) as src:
    # src.res returns a tuple (pixel_width, pixel_height)
    pixel_width = src.res[0] 
    print(f"The Pixel Size (GSD) is approximately: {pixel_width} meters per pixel")

