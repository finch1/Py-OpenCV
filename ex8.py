# Date: 29/05/2018
# Box Filter - Blur

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def plot_cv_img(input_image, output_image):
    # converts an image from BGR to RGB and plots
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Box Filter (5,5)')
    ax[1].axis('off')

    plt.show()

def main():
    # read an image
    img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')

    # to try different kernel, change size here
    kernel_size = (25,25)

    # opencv has implementation for kernel based box blurring
    blur = cv2.blur(img, kernel_size)

    # do plot
    plot_cv_img(img, blur)

    # applying a box filter with hard edges doesn't result in a smooth blur on the output
    # to improve this, the filter can be made smoother around the edges with Gaussian filter
    # This is a non linear filter which enhances the effect of the center pixel and 
    # grafually reduces the effects as the pixel gets farther from the center
    #      1  4  6  4 1
    #      4 16 24 16 4
    #   1  6 24 36 24 6
    #   _  4 16 24 16 4
    #  256 1  4  6  4 1

    # apply gaussian blur, kernel of size 5x5
    # sigma values are same in both direction
    g_blur = cv2.GaussianBlur(img, kernel_size, 0)
    plot_cv_img(blur, g_blur)


if __name__ == '__main__':
    main()
