INPUT_IMAGE = "data/enhanced_data/enhanced.png"
INPUT_TIF = "data/raw_data/sample.tif"
# OUTPUT_TIF_TILES_DIR = "data/tif_tiles_512_128"
OUTPUT_TIF_TILES_DIR = "data/tif_tiles_1024_128"
OUTPUT_PNG_TILES_DIR = "data/tif_tiles_png_1024_128"



SMALL_IMAGES_DIR = "data/enhanced_topographic_tiles_512_128"
SEGMENTATION_MODEL = "models/sam3.pt"

SEGMENTATION_OUTPUT_DIR = "data/segmentation_output_v2"
OUTPUT_GEMINI_IMAGES_DIR = "data/gemini_image_tiles"
OUTPUT_AUGMENTED_IMAGES_DIR = "data/augmented_images"
# OUTPUT_AUGMENTED_IMAGES_DIR = "data/augmented_small_images"
IMAGE_TILES_DIR = "data/image_tiles_256"

JSON_PATH = "data/tif_tiles_png_1024_128/Eval/annotations.json"

TILE_SIZE = 1024
OVERLAP = 128

ENHANCED_IMAGE = "data/enhanced_data/enhanced.tif"


# Image Params
BRIGHTNESS = (0.05,0.05)        
EXPOSURE = 0.35            
CONTRAST = (0.1,0.1)            
HIGHLIGHTS = 0.10          
SHADOWS = -0.10           
SATURATION = 0.10          
SHARPNESS = (0.05, 0.05)           