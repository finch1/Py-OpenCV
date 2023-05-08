## converting between color spaces

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import cv2

def plot_img(input_image):
    """
    Takes in image
    Plots image using matplotlib
    """
    # change color channel order for marplotlib
    plt.imshow(input_image)

    # for easier view, turn off axis around image
    plt.axis('off')
    plt.show() 


# loads and reads an image from path to file
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')  

# convert the color to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert the color to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# convert the color to hls
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# convert the color to lab and back to bgr
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
bgr = cv2.cvtColor(img, cv2.COLOR_LAB2BGR) 

# disp img
cv2.imshow("Image Gray", gray)
cv2.imshow("Image HSV", hsv)
cv2.imshow("Image HLS", hls)
cv2.imshow("Image LAB", lab)
cv2.imshow("Image BGR", bgr)
plot_img(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
       
""" 
LAB color space: L: lightness A: green-red colors B: blue-yellow colors
This is used to convert between one type of color space (RGB) to others
such as (CMYK).
"""