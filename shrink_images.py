import os

import numpy
import matplotlib.pyplot as plt
from read_roi import read_roi_zip
from PIL import Image, ImageDraw

data_folder = "data/Animal 1 Photomerge/"
target_size = (624, 624)

resize_folder = "resized/Animal 1 Photomerge (624 x 624)/"
os.makedirs(resize_folder, exist_ok=True)

images = []
if(os.path.isdir(data_folder)):
    print("Valid Data Folder")
    files = os.listdir(data_folder)
    num_files = len(files)
    print("Files found: " + str(num_files))
    for idx, file_path in enumerate(files):
        print("File " + str(idx+1) + " of " + str(num_files))
        if(file_path.endswith(".tif")):
            print("Resizing " + file_path)
            im = Image.open(data_folder+file_path)
            im = im.resize(target_size)
            im.save(resize_folder + file_path)


            

    