# Date: 01/06/2018
# Image Gradient

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# read an image
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')


# sobel
#x_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0 ksize = 5)
#y_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1 ksize = 5)

# laplacian
lapl = cv2.Laplacian(img, cv2.CV_64F, ksize = 5)

# gaussian blur
blur = cv2.GaussianBlur(img,(5,5),0)
# laplacian of gaussian
log = cv2.Laplacian(blur, cv2.CV_64F, ksize = 5)
