# Date: 01/06/2018
"""
ORB Features
First create an ORB onject and update parameter values:
    orb = cv2.ORB_create()
    # set parameters
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

Detect keypoints from previously created ORB object:
    # detect keypoints
    kp = orb.detect(img, None)

Lastly, compute descriptors from each keypoints detected:
    # for detected keypoints compute decriptors
    kp, des = orb.compute(img, kp)
"""

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

def compute_orb_keypoints(filename):
    # read image from filename and computes ORB keypoints
    # returns image, keypoints and descriptors

    # read an image
    img = cv2.imread(filename)
    # create orb object
    orb = cv2.ORB_create()

    # set parameters
    # FAST feature type
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    # detect keypoints
    kp = orb.detect(img, None)

    # for detected keypoints compute descriptors
    kp, des = orb.compute(img, kp)

    return img, kp, des

def draw_keyp(img, kp):
    # takes image and keypoints and plots on the same images. does not display it
    cv2.drawKeypoints(img, kp, img, color=(255,0,0), flags=2)
    return img

def plot_img(img, figsize=(12,8)):
    # plots image using matplotlib for the given figsize

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)

    # image need to be converted to RGB format for plotting 
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def main():
    # image path
    filename = 'C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png'
    # compute ORB keypoints
    img1, kp1, des1 = compute_orb_keypoints(filename)
    # draw keypoints on image
    img1 = draw_keyp(img1, kp1)
    # plot image with keypoints
    plot_img(img1)

if __name__ == "__main__":
    main()
