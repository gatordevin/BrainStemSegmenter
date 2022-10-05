import os
from PIL import Image, ImageDraw
import numpy
import matplotlib.pyplot as plt
from read_roi import read_roi_zip

data_folder = "data/Animal 1/Slide A/NEUN"

image_label_pair = []
if(os.path.isdir(data_folder)):
    print("Valid Data Folder")
    files = os.listdir(data_folder)
    image_paths = []
    label_paths = []
    for file_path in files:
        if(file_path.endswith(".tif")):
            image_paths.append(file_path)
        elif(file_path.endswith("_roi.zip")):
            label_paths.append(file_path)
    
    temp_image_paths = image_paths
    for image_path in temp_image_paths:
        image_name = image_path.replace(".tif", "")
        found_label = False
        for label_path in label_paths:
            label_name = label_path.replace("_roi.zip", "")
            if(label_name==image_name):
                found_label = True
                break
        if(not found_label):
            image_paths.remove(image_path)

    temp_label_paths = label_paths
    for label_path in temp_label_paths:
        label_name = label_path.replace("_roi.zip", "")
        found_image = False
        for image_path in image_paths:
            image_name = image_path.replace(".tif", "")
            if(label_name==image_name):
                found_image = True
                break
        if(not found_image):
            label_paths.remove(label_path)
    image_label_pair = list(zip(image_paths, label_paths))

print("Image Label pairs found : " + str(image_label_pair))

processed_data = "processed/"
if(not os.path.isdir(processed_data)):
    os.makedirs(processed_data)

dataset_index = len(os.listdir(processed_data))
dataset_folder_name = "dataset_" + str(dataset_index)
os.makedirs(processed_data + dataset_folder_name)

for idx, (image_path, label_path) in enumerate(image_label_pair):
    im = Image.open(data_folder + "/" + image_path).convert('L')
    imarray = numpy.array(im)
    rois = read_roi_zip(data_folder + "/" + label_path)
    size = im.size
    shape = imarray.shape
    # print("Image shape : " + str(shape))

    label_im = Image.new('L', size, 0)
    for pidx, label in enumerate(rois.keys()):
        x_points = rois[label]["x"]
        y_points = rois[label]["y"]
        polygon = list(zip(x_points, y_points))
        ImageDraw.Draw(label_im).polygon(polygon, outline=pidx+1, fill=pidx+1)
    
    mask = numpy.array(label_im)
    im.save(processed_data + dataset_folder_name + "/image_" + str(idx) + ".png", 'PNG')
    label_im.save(processed_data + dataset_folder_name + "/image_" + str(idx) + "_label.png", 'PNG')
    print("image_" + str(idx) + " saved")
