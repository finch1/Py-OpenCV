## plot zyx for grayscale intensity
## display image

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import cv2

# loads and reads an image from path to file
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')    # Change Flag As 1 For Color Image
                                                                                    # or O for Gray Image So It image is 
                                                                                    # already gray
#convert the color to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# resize the image(optimal)
gray = cv2.resize(gray, (160, 120))

# apply smoothing operation
gray = cv2.blur(gray, (3,3))

# create grid to plot using numpy
xx, yy = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xx, yy, gray, rstride=1, cstride=1, cmap='gray', linewidth=1)

#show
plt.show() 

#display image
cv2.imshow("Image", img)

# keeps the window open until a key is pressed
cv2.waitKey(0)

# clears all window buffers
cv2.destroyAllWindows()