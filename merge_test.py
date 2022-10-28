from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy
import imutils

def openImages(image_paths):
    imgs = []
    for img_path in image_paths:
        im = Image.open(img_path).convert('L')
        img = numpy.array(im)
        imgs.append(img)
    return imgs

def trim(frame):
    y_nonzero, x_nonzero = numpy.nonzero(frame)
    return frame[numpy.min(y_nonzero):numpy.max(y_nonzero), numpy.min(x_nonzero):numpy.max(x_nonzero)]

def stitch_images(imageOne, imageTwo):
    f, axarr = plt.subplots(2,2)
    axarr[0][0].imshow(imageOne*3, cmap="gray")
    axarr[1][0].imshow(imageTwo*3, cmap="gray")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imageOne, None)
    kp2, des2 = sift.detectAndCompute(imageTwo, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m in matches:
        if(m[0].distance <0.5*m[1].distance):
            good.append(m)
    matches = numpy.asarray(good)

    if(len(matches[:,0]) >= 4):
        src = numpy.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = numpy.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    imageOneMask = numpy.zeros(imageOne.shape, dtype=numpy.uint8)
    imageOneMask[:] = 255
    

    dst = cv2.warpPerspective(imageOne, H, ((imageOne.shape[1]+imageTwo.shape[1]),(imageOne.shape[0]+imageTwo.shape[0])))
    mask = cv2.warpPerspective(imageOneMask, H, ((imageOne.shape[1]+imageTwo.shape[1]),(imageOne.shape[0]+imageTwo.shape[0])))
    mask = cv2.bitwise_not(mask)
    axarr[0][1].imshow(dst*3, cmap="gray")
    
    new_image = numpy.zeros((dst.shape[0],dst.shape[1]),dtype=numpy.uint8)
    new_image[0:imageTwo.shape[0], 0:imageTwo.shape[1]] = imageTwo
    
    new_image = cv2.bitwise_and(new_image,mask)
    new_image = cv2.bitwise_or(new_image,dst)
    # axarr[1][1].imshow(new_image*3, cmap="gray")
    # dst[0:imageTwo.shape[0], 0:imageTwo.shape[1]] = imageTwo
    # FIX ADDING NEED TO ADD ONLY PIXELS NOT BLACK
    
    new_image = trim(new_image)
    axarr[1][1].imshow(new_image*3, cmap="gray")
    # plt.imshow(dst*3, cmap="gray")
    plt.show()
    
    return dst

def stitch_all(images):
    baseImage = images[0]
    for image in images[1:]:
        baseImage = stitch_images(image, baseImage)

image_paths = [
    "NEUN/sAc2r3.1NEUN.tif",
    "NEUN/sAc2r3.2NEUN.tif",
    "NEUN/sAc2r3.3NEUN.tif",
    "NEUN/sAc2r3.4NEUN.tif",
    "NEUN/sAc2r3.5NEUN.tif",
    "NEUN/sAc2r3.6NEUN.tif",
]

images = openImages(image_paths)
stitch_all(images)
# print("images opened")
# mergeAllImage(images)
