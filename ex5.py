# Date: 28/05

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# a plotting function for a colored image read in OpenCV
def plot_cv_img(input_image):
    # converts an image from bgr to rgb plots
    rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # converts an image from rgb to gray plots
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # this will plot the image thinking the format is RGB
    rgbinvert = input_image 
    # take out a patch of pixels
    patch_gray = gray[250:260, 250:260]

    plt.subplot(221)
    plt.imshow(rgb)
    plt.subplot(222)
    plt.imshow(gray, cmap='gray')
    plt.subplot(223)
    plt.imshow(rgbinvert)
    plt.subplot(224)
    plt.imshow(patch_gray, cmap='gray')
    plt.axis("off")
    plt.show()


# read image
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')
plot_cv_img(img)
