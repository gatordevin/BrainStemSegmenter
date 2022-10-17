import cv2
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from time import sleep
import numpy

directory = "NEUN/"
save_dir = "photomerged/"
os.makedirs(save_dir, exist_ok=True)
sections = {}
for path in os.listdir(directory):
    if(path.endswith("tif")):
        current_section = path.split(".")[0]
        if(current_section!="" or current_section!=" " or "PM" not in current_section):
            if(current_section not in sections.keys()):
                current_paths = []
                for section_path in os.listdir(directory):
                    if current_section in section_path:
                        # print(section_path)
                        current_paths.append(section_path)
                sections[current_section] = current_paths
print(sections)
        # print(current_paths)

def merge_images(sections):
    stitcher = cv2.Stitcher_create()
    for image_name in sections.keys():
        image_name = "sAc2r4"
        print("Stiching: " + image_name)
        images = []
        number_of_images = len(sections[image_name]) 
        for i in range(number_of_images):
            image_path = ""
            if("NEUN" in sections[image_name][0]):
                image_path = directory + image_name + "." + str(i + 1) + "NEUN.tif"
            else:
                image_path = directory + image_name + "." + str(i + 1) + "n.tif"
            image = cv2.imread(image_path)
            images.append(image)
            print(image_path)
        sleep(3)
        print(len(images))
        (status, stitched) = stitcher.stitch(images)
        if(status!=0):
            print("trying reversed")
            images.reverse()
            sleep(3)
            (status, stitched) = stitcher.stitch(images)
        print(status)
        img = Image.fromarray(stitched, "RGB").convert("L")
        img.save(save_dir + image_name + " PM NEUN.png", 'PNG')
        img.save(save_dir + image_name + " PM NEUN.tif", 'TIFF')
        # plt.imshow(stitched*2)
        # plt.show()

# sections["sAc2r4"] = [
#     # "NEUN/sAc2r3.1NEUN.tif",
#     # "NEUN/sAc2r3.2NEUN.tif",
#     # "NEUN/sAc2r3.3NEUN.tif",
#     "NEUN/sAc2r4.1NEUN.tif",
#     "NEUN/sAc2r4.2NEUN.tif",
#     "NEUN/sAc2r4.3NEUN.tif",
#     "NEUN/sAc2r4.4NEUN.tif",
#     "NEUN/sAc2r4.5NEUN.tif",
#     "NEUN/sAc2r4.6NEUN.tif",
#     "NEUN/sAc2r4.7NEUN.tif",
# ]

merge_images(sections)

# sections["sAc2r3"] = [
#     "photomerged/sAc2r3 PM NEUN.tif",
#     "NEUN/sAc2r3.6NEUN.tif",
# ]

# merge_images(sections)

# imagePaths = [
#     "NEUN/sAc1r2.1NEUN.tif",
#     "NEUN/sAc1r2.2NEUN.tif",
#     "NEUN/sAc1r2.3NEUN.tif",
#     "NEUN/sAc1r2.4NEUN.tif",
#     "NEUN/sAc1r2.5NEUN.tif",
#     "NEUN/sAc1r2.6NEUN.tif",
#     "NEUN/sAc1r2.7NEUN.tif",
#     "NEUN/sAc1r2.8NEUN.tif",
#     "NEUN/sAc1r2.9NEUN.tif",
#     "NEUN/sAc1r2.10NEUN.tif",
#     "NEUN/sAc1r2.11NEUN.tif",
#     "NEUN/sAc1r2.12NEUN.tif",
# ]
# # imagePaths = [
# #     "NEUN/sAc1r3.1NEUN.tif",
# #     "NEUN/sAc1r3.2NEUN.tif",
# #     "NEUN/sAc1r3.3NEUN.tif",
# #     "NEUN/sAc1r3.4NEUN.tif",
# #     "NEUN/sAc1r3.5NEUN.tif",
# #     "NEUN/sAc1r3.6NEUN.tif",
# #     "NEUN/sAc1r3.7NEUN.tif",
# #     "NEUN/sAc1r3.8NEUN.tif",
# # ]

# # imagePaths = [
# #     "NEUN/sAc1r4.1NEUN.tif",
# #     "NEUN/sAc1r4.2NEUN.tif",
# #     "NEUN/sAc1r4.3NEUN.tif",
# #     "NEUN/sAc1r4.4NEUN.tif",
# #     "NEUN/sAc1r4.5NEUN.tif",
# #     "NEUN/sAc1r4.6NEUN.tif",
# #     "NEUN/sAc1r4.7NEUN.tif",
# #     "NEUN/sAc1r4.8NEUN.tif",
# #     "NEUN/sAc1r4.9NEUN.tif",
# #     "NEUN/sAc1r4.10NEUN.tif",
# # ]
# images = []
# for imagePath in imagePaths:
#     # im = Image.open(imagePath)
#     image = cv2.imread(imagePath)
#     images.append(image)
#     # plt.imshow(image)
#     # plt.show()
    
# sleep(1)
# stitcher = cv2.Stitcher_create()
# (status, stitched) = stitcher.stitch(images)
# print(status)
# plt.imshow(stitched*2)
# plt.show()