# Date: 29/05/2018
# Linear Filters. Point Based Filter

# K changes brightness
# L changes contrast
# f(i,j) is the image
# g(i,j) is the new image
# g(i,j) = K x f(i,j) + L

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def point_op(img, K, L):
    # applies point operation to given grayscale image
    img = np.asarray(img, dtype=np.float)
    img = img*K + L
    # clip pixel values
    img[img > 255] = 255
    img[img < 0] = 0
    return np.asarray(img, dtype= np.int)

def main():
    # read an image
    img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')
    plt.subplot(221)
    plt.title("GRAY")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray', interpolation='nearest', aspect='auto')

    # k = 0.5, l = 0
    k = 0.5
    l = 0
    plt.subplot(222)
    plt.title("B = 0.5, C = 0")
    out1 = point_op(gray, k, l)
    plt.imshow(out1, cmap='gray', interpolation='nearest', aspect='auto')

    # k = 1., l = 10
    k = 3.
    l = 10
    plt.subplot(223)
    plt.title("B = 1., C = 10")
    out2 = point_op(gray, k, l)
    plt.imshow(out2, cmap='gray', interpolation='nearest', aspect='auto')

    #k = 0.8., l = 15
    k = 0.8
    l = 15
    plt.subplot(224)
    plt.title("B = 0.8, C = 15")
    out3 = point_op(gray, k, l)
    plt.imshow(out3, cmap='gray', interpolation='nearest', aspect='auto')
    plt.show()

    plt.subplot(221)
    plt.title("PATCH GRAY")
    patch_gray = gray[500:600, 340:400]
    plt.imshow(patch_gray, cmap='gray', interpolation='nearest', aspect='auto')

    plt.subplot(222)
    plt.title("PATCH B = 0.5, C = 0")
    patch_out1 = out1[500:600, 340:400]
    plt.imshow(patch_out1, cmap='gray', interpolation='nearest', aspect='auto')

    plt.subplot(223)
    plt.title("PATCH B = 1., C = 10")
    patch_out2 = out2[500:600, 340:400]
    plt.imshow(patch_out2, cmap='gray', interpolation='nearest', aspect='auto')

    plt.subplot(224)
    plt.title("PATCH B = 0.8, C = 15")
    patch_out3 = out3[500:600, 340:400]
    plt.imshow(patch_out3, cmap='gray', interpolation='nearest', aspect='auto')
    
    # res = np.hstack([gray, out1, out2, out3])
    plt.show()

if __name__ == '__main__':
    main()

    