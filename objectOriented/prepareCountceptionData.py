import os
from skimage import io
from scipy import stats
from roifile import ImagejRoi
from matplotlib import pyplot as plt
import numpy as np

dataset_dir = "C:/Users/gator/BrainStemSegmenter/Data_10-28-2022"
crop_image_shape = (300, 300)
patch_size = 32
noutputs = 1

def remove_coord(target_coord, coord_list):
    for idx, coord in enumerate(coord_list):
        if(target_coord[0]==coord[0]):
            if(target_coord[1]==coord[1]):
                coord_list = np.delete(coord_list, idx, 0)
    return coord_list

# def genGausImage(framesize, mx, my, cov=1):
#     x, y = np.mgrid[0:framesize, 0:framesize]
#     pos = np.dstack((x, y))
#     mean = [mx, my]
#     cov = [[cov, 0], [0, cov]]
#     rv = stats.multivariate_normal(mean, cov).pdf(pos)
#     return rv/rv.sum()

# def getDensity(width, markers):
#     gaus_img = np.zeros((width,width))
#     for k in range(width):
#         for l in range(width):
#             if (markers[k,l] > 0.5):
#                 gaus_img += genGausImage(len(markers),k-patch_size/2,l-patch_size/2,cov)
#     return gaus_img

# def getMarkersCells(rois, size):  
#     lab = io.imread("objectOriented/TestPics/BM_GRAZ_HE_0001_01_000.png")[:,:, 0]/255
#     # lab = np.zeros(size) 
#     # print(lab.shape)
#     # for roi in rois:
#     #     lab[int(roi[1]),int(roi[0])] = 1
#     print(lab.shape)
#     binsize = [2,2]
#     out = np.zeros(size)
#     for i in range(binsize[0]):
#         for j in range(binsize[1]):
#             print(i, j)
#             print(lab[i::binsize[0], j::binsize[1]].shape)
#             out = np.maximum(lab[i::binsize[0], j::binsize[1]], out)
        
#     print(lab.sum(),out.sum())
#     # assert np.allclose(lab.sum(),out.sum(), 1)
    
    
#     return out#np.pad(lab,patch_size, "constant")

# def getCellCountCells(markers, size):
#     x,y,h,w = size
#     types = [0] * noutputs
#     for i in range(noutputs):
#         types[i] = (markers[y:y+h,x:x+w] == 1).sum()
#         #types[i] = (markers[y:y+h,x:x+w] != -1).sum()
#     return types

# def getLabelsCells(markers, img_pad, stride, scale):
#     height = int((img_pad.shape[0])/stride)
#     width = int((img_pad.shape[1])/stride)
#     print("label size: ", height, width)
#     labels = np.zeros((noutputs, height, width))
#     print("base_x",0, 0, height, height)
#     for y in range(0,height):
#         for x in range(0,width):
#             count = getCellCountCells(markers,(x*stride,y*stride,patch_size,patch_size))  
#             for i in range(0,noutputs):
#                 labels[i][y][x] = count[i]
    

#     count_total = getCellCountCells(markers,(0,0,height+patch_size,width+patch_size))
#     return labels, count_total

# def getTrainingExampleCells(img, framesize_w, framesize_h, rois, stride, scale):
#     label_img_shape = (framesize_w+patch_size, framesize_h+patch_size)
#     markers = getMarkersCells(rois, (300, 300))
#     markers = markers[0:framesize_h, 0:framesize_w]
#     markers = np.pad(markers, patch_size, "constant", constant_values=-1)
    
#     labels, count  = getLabelsCells(markers, rois, stride, scale)
#     return img, labels, count

# image = io.imread("objectOriented/TestPics/BM_GRAZ_HE_0001_01_000.png")
# img, lab, count = getTrainingExampleCells("objectOriented/TestPics/BM_GRAZ_HE_0001_01_dots_000.png", crop_image_shape[0], crop_image_shape[1], [], 1, 2)

for file_name in os.listdir(dataset_dir):
    file_path = dataset_dir + "/" + file_name
    image_name = file_name.replace(".roi", ".tif")
    cropped_path = dataset_dir + "/cropped"
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

            label_img = np.zeros((crop_image.shape[0]+patch_size, crop_image.shape[1]+patch_size))
            for x, y in rois_in_crop:
                left = int(x)
                top = int(y)
                right = int(x+patch_size)
                bottom = int(y+patch_size)
                label_img[top:bottom,left:right] += 1

            # Save images to folder
            split_image_name = image_name.split(".")
            cropped_file_name = split_image_name[0] + "_cropped_count_" + str(len(rois_in_crop)) + "_" + str(index) + ".png"
            cropped_label_file_name = split_image_name[0] + "_cropped_label_count_" + str(len(rois_in_crop)) + "_" + str(index) + ".png"
            cropped_file_path = cropped_path + "/" + cropped_file_name
            cropped_label_file_path = cropped_path + "/" + cropped_label_file_name
            io.imsave(cropped_file_path, crop_image)
            io.imsave(cropped_label_file_path, label_img)
            print("saved: " + cropped_label_file_name)
            print("saved: " + cropped_file_name)
            index+=1

