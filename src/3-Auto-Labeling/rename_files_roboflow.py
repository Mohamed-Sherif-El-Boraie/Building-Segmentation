import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import json
import os

image_dir = 'data/annotated_data/train/'
# Rename image files
# Rename image files
for file in os.listdir(image_dir):
    if file.endswith('.jpg') and 'tile_' in file:
        tile_num = file.split('_')[1]
        new_name = f"tile_{tile_num}.png"
        
        old_path = os.path.join(image_dir, file)
        new_path = os.path.join(image_dir, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

# Update JSON file
json_path = os.path.join(image_dir, '_annotations.coco.json')

with open(json_path, 'r') as f:
    data = json.load(f)

for img in data.get('images', []):
    old_name = img['file_name']
    if 'tile_' in old_name:
        tile_num = old_name.split('_')[1]
        img['file_name'] = f"tile_{tile_num}.png"
        print(f"Updated JSON: {old_name} -> {img['file_name']}")

with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)
