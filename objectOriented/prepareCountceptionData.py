import os
from skimage import io
from scipy import stats
from roifile import ImagejRoi
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
import cv2

dataset_dir = "C:/Users/gator/FullerLab/BrainStemSegmenter/Data_10-28-2022"
crop_image_shape = (300, 300)
image_scale = 1
patch_size = 32
noutputs = 1

def remove_coord(target_coord, coord_list):
    for idx, coord in enumerate(coord_list):
        if(target_coord[0]==coord[0]):
            if(target_coord[1]==coord[1]):
                coord_list = np.delete(coord_list, idx, 0)
    return coord_list

for file_name in os.listdir(dataset_dir):
    file_path = dataset_dir + "/" + file_name
    image_name = file_name.replace(".roi", ".tif")
    cropped_path = dataset_dir + "/cropped_half_scale"
    os.makedirs(cropped_path, exist_ok=True)
    if ".roi" in file_path:
        roi = ImagejRoi.fromfile(file_path)
        image_path = file_path.replace(".roi", ".tif")
        image = io.imread(image_path)
        image_shape = image.shape

        all_rois = roi.coordinates()
        rois_remaining = roi.coordinates()
        index = 0
        while len(rois_remaining)>0:
            print(len(rois_remaining))
            x, y = rois_remaining[0]
            left = int(x-(crop_image_shape[0]/2))
            top = int(y-(crop_image_shape[1]/2))
            right = int(x+(crop_image_shape[0]/2))
            bottom = int(y+(crop_image_shape[1]/2))
            crop_image = image[top:bottom,left:right]
            print(crop_image.shape)
            rois_in_crop = []
            for roi in all_rois:
                roi_x, roi_y = roi
                if(top<roi_y<bottom and left<roi_x<right):
                    rois_in_crop.append([roi_x-left, roi_y-top])
                    rois_remaining = remove_coord(roi, rois_remaining)
            rois_in_crop = np.array(rois_in_crop)

            label_img = np.zeros((crop_image.shape[0]+(patch_size*image_scale), crop_image.shape[1]+(patch_size*image_scale))).astype(np.uint8)
            for x, y in rois_in_crop:
                left = int(x)
                top = int(y)
                right = int(x+(patch_size*image_scale))
                bottom = int(y+(patch_size*image_scale))
                label_img[top:bottom,left:right] += 1
            
            crop_image = cv2.resize(crop_image, dsize=(int(crop_image.shape[0]/image_scale), int(crop_image.shape[1]/image_scale)), interpolation=cv2.INTER_CUBIC)
            label_img = cv2.resize(label_img, dsize=(int(label_img.shape[0]/image_scale), int(label_img.shape[1]/image_scale)), interpolation=cv2.INTER_CUBIC)

            # Save images to folder
            split_image_name = image_name.split(".")
            cropped_file_name = split_image_name[0] + "_cropped_count_" + str(len(rois_in_crop)) + "_" + str(index) + ".png"
            cropped_label_file_name = split_image_name[0] + "_cropped_label_count_" + str(len(rois_in_crop)) + "_" + str(index) + ".png"
            cropped_file_path = cropped_path + "/" + cropped_file_name
            cropped_label_file_path = cropped_path + "/" + cropped_label_file_name
            io.imsave(cropped_file_path, crop_image)
            io.imsave(cropped_label_file_path, label_img)
            f, axarr = plt.subplots(2,1)
            # axarr[0].imshow(label_img.astype(np.uint8))
            # axarr[1].imshow(crop_image.astype(np.uint8))
            # plt.show()
            print("saved: " + cropped_label_file_name)
            print("saved: " + cropped_file_name)
            index+=1

