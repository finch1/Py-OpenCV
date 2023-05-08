# Date: 29/05/2018
# Histogram Equilization
"""
Using a histogram equilization technique, changes in brightness and contrast can be found 
algorithmically and creates a better looking photo. Intuitively, this method tries to set
the brightest pixels to white and darker pixels to black. The remaining pixel values are 
similarly rescaled. This rescaling is performed by transforming original intensity
distribution to caputre all intensity distribution. 

"""

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def plot_cv_img(input_image, output_image):
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(input_image, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(output_image, cmap='gray')
    ax[1].set_title('Hist Equ')
    ax[1].axis('off')
    # plt.savefig()
    plt.show()

def main():
    # read an image
    img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform equilization on the input image
    equ = cv2.equalizeHist(gray)
    plot_cv_img(gray, equ)

    # for c in range(0, 2):
    #     equ[:,:,c] = cv2.equalizeHist(img[:,:,c])
    
    # hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf  * hist.max() / cdf.max()
    # plt.plot(cdf_normalized, color = 'b')
    # plt.hist(img.flatten(), 256, [0,256], color='r')
    # plt.xlim([0,256])
    # plt.legend(('cdf', 'histogram'), loc = 'upper left')
    # plt.show()

if __name__ == "__main__":
    main()