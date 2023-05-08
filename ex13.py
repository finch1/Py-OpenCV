# Date: 01/06/2018
# FAST Featres from Accelerated Segment Test

"""
The algorithm uses pixel neighbourhood to compute key points in an image. 

1. Initialize detector using cv2.FastFeatureDetector_create()
2. Setup threshold parameters for filtering detections
3. Setup flag if non-maximal suppression to be used for clearing neighbourhood regions
   of repeated detections
4. Detect keypoints and plot them on the input image
"""

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# read an image
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')

"""
Reads image from filename and computes FAST keypoints.
Returns image with keypoints
is_nms: flag to use Non-maximal suppression
thresh: threshold value
"""

is_nms = True
thresh = 40

# initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find and draw the keypoints
if not is_nms:
    fast.setNonmaxSuppression(0)

    fast.setThreshold(thresh)
    kp = fast.detect(img, None)
    cv2.drawKeypoints(img, kp, color=(255, 0, 0))

plt.imshow(img, cmap='gray')
plt.show()
