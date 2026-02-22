## Setup

### Creating venv with uv
```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate

# install and sync deps  
uv init
uv add [deps] 
# compile from pyproject.toml
uv pip compile pyproject.toml -o requirements.txt
```
.venv\Scripts\activate





# SAM 3 Repo installation
Install PyTorch with CUDA support:
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Clone the repository and install the package:
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

# Download SAM3.pt from huggingface
```bash
sam3.pt
```

# Description in High Level
- this project is a pipeline for building segmentation using SAM3 (Segment Anything Model) 

# Project Segmentation Flow

1. **Preprocessing**:   `convert_png.py` - Convert the tif image to RGB png. 
                        `image_divider.py` - Split the png image into smaller images of size 256x256 . 
                                         - Augment the images using albumentations. 
                                         - Save the augmented images in the same directory. 

2. **Preprocessing**:    We choose the best preprocessing method between 3 files:
                        1. `DLSR_pipeline.py` - Deep Learning Super-Resolution Model (DLSR) using OpenCV pre-trained neural network to do upscaling and enhance the quality of the images. EDSR / FSRCNN
                        2. `preprocess_satellite_gpt.py` - Enhances satellite GeoTIFF images through contrast normalization and adaptive brightness optimization, delivering clearer and more consistent imagery.
                        3. `preprocess_satellite_gemini.py` - Automates the preprocessing of raw satellite TIFs by extracting the RGB bands and applying Contrast Limited Adaptive Histogram Equalization (CLAHE) to reveal hidden details in varied lighting. 
                                    
3. **Auto-Labeling**:   `sam3_semantic_predictor.py` - Auto-labeling using SAM3 (Segment Anything Model) to label the images.
                                                     - Extract the coco json annotations from the images.
                                                     - Save the coco json annotations in the same directory.
                                                    
4. **Postprocessing**:  1- `coordinate_mapping.py` - Merge Split Building Segmentations Across Tiles. 
                        Buildings in the middle of tiles are preserved exactly as detected.
                        2- `downscaling.py` - Downscale the coco json annotations to the real size of the image. 
                            2.1 - `plot_tif.py` - Plot the results of the segmentation on the original tif image using the COCO downscaled json annotations.
                            2.2 - `coco_to_esri_json.py` - Convert COCO downscaled json annotations to Esri JSON format compatible with ArcGIS and other Esri products. 




install from sam3 uv pip install -e .\models\sam3\