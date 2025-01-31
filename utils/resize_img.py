# This file is used to resize subject images to the same size
from PIL import Image
import os

directory = "../images/dreambooth_dataset/dog6"
output_dir = "../exp/resized_imgs"

for filename in os.listdir(directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        filepath = os.path.join(directory, filename)
        with Image.open(filepath) as img:
            img = img.resize((512, 512))
            img.save(os.path.join(output_dir, filename))