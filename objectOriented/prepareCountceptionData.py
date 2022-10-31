import os
from skimage import io
from roifile import ImagejRoi
from matplotlib import pyplot as plt
import numpy as np

dataset_dir = "C:/Users/gator/FullerLab/BrainStemSegmenter/Data_10-28-2022"
crop_image_shape = (300, 300)

def remove_coord(target_coord, coord_list):
    for idx, coord in enumerate(coord_list):
        if(target_coord[0]==coord[0]):
            if(target_coord[1]==coord[1]):
                coord_list = np.delete(coord_list, idx, 0)
    return coord_list

for file_name in os.listdir(dataset_dir):
    file_path = dataset_dir + "/" + file_name
    # print(file_path)
    if ".roi" in file_path:
        roi = ImagejRoi.fromfile(file_path)
        image_path = file_path.replace(".roi", ".tif")
        image = io.imread(image_path)
        image_shape = image.shape

        all_rois = roi.coordinates()
        rois_remaining = roi.coordinates()
        while len(rois_remaining)>0:
            print(len(rois_remaining))
            x, y = rois_remaining[0]
            left = int(x-150)
            top = int(y-150)
            right = int(x+150)
            bottom = int(y+150)
            crop_image = image[top:bottom,left:right]

            rois_in_crop = []
            for roi in all_rois:
                roi_x, roi_y = roi
                if(top<roi_y<bottom and left<roi_x<right):
                    rois_in_crop.append([roi_x-left, roi_y-top])
                    rois_remaining = remove_coord(roi, rois_remaining)
            rois_in_crop = np.array(rois_in_crop)
            # Need to take rois_in_crop and produce heatmap then output heatmap and crop pair to a folder in this directory with same name
            # We the need to apply all our trasnforms to the iamges and save them to disk to reduce processing time during training
            plt.plot([x for x,y in rois_in_crop], [y     for x,y in rois_in_crop], marker='v', color="white")
            
            plt.imshow(crop_image)
            plt.show()

