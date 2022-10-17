import cv2
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from time import sleep
import numpy

folder = "C:/Users/gator/Downloads/Photos-001"

images = []
for path in os.listdir(folder):
    image = cv2.imread(folder + "/" + path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

plt.imshow(stitched)
plt.show()

img = Image.fromarray(stitched, "RGB")
img.save("C:/Users/gator/Downloads/farm.png", 'PNG')