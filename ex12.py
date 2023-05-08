# Date: 01/06/2018

""" 
In general, an image matching procedure works as follows:

1. Extract robust features from a given image. This involves searching through 
the whole image for possible features and then thresholding them. There are 
several techniques for the selection of features. The feature extracted, in 
come cases, needs to be converted into a more descriptive form such that 
it is learnt by the model or can be stored for re-reading.

2. In the case of feature matching, we are given a sample image and we would
like to see whether this matches a reference image. After feature detection
and extraction, as shown previously, a distance metric is formed to compute
the distance between features of a sample with respect to the features of 
reference. If this distance is less than the threshold, we can say the 2 
images are similar. 

3. For feature tracking, we omit previously explained feture matching steps.
Instead of globally matching features, the focus is more on neighborhood 
matching. This is used in cases such as image stabilization, object tracking,
or motion detection. 
"""
# Harris Corner Visualization
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# read an image
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# harris corner parameters
block_size = 4 # covariance matrix size
kernel_size = 3 # neighbourhood kernel
k = 0.01 # parameter for harris corner score

# compute harris corner
corners = cv2.cornerHarris(gray, block_size, kernel_size, k)

# create corner image
display_corner = np.ones(gray.shape)
print(gray.shape)
display_corner = display_corner*255

# apply thresholding to the corner score
thres = 0.01 # more then 1% of max value
display_corner[corners>thres*corners.max()] = 10 # display value

# set up display
plt.figure(figsize=(12,8))
plt.imshow(display_corner, cmap='gray')
plt.show()