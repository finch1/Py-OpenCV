# Date: 29/05/2018
# random noies filter
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def plot_cv_img(input_image):
    # initiaze gray scale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    x, y = gray.shape
    # initialize noise image with zeros
    noise = np.zeros((x, y))

    # fill the image with random numbers in given range
    cv2.randu(noise, 0, 256)
    # add noise to existing image
    # the 0.2 parameter increase or decrease the value to create different intensity noise
    noise_gray = gray + np.array(0.2*noise, dtype=np.int)
    plt.imshow(noise_gray, cmap='gray')
    plt.show()

# read image
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')
plot_cv_img(img)