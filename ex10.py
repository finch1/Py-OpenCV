# Date: 29/05/2018
# median filter
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# read an image
img = cv2.imread('C:\\Users\\GOKU\\Documents\\PyLrn\\Python OpenCV\\public.png')

# initialize noise image with zeros
# x[:2] is short for x[0:2] which gibes a 'slice' ranging from entry 0 to 1 ex: x=(0,1,2,3) out: (0,1)
noise = np.zeros(img.shape[:2])
print(noise)
# fill the image with random numbers in given range
cv2.randu(noise, 0, 256)

# add noise to existing image, apply channel wise

noise_factor = 0.1
noisy_img = np.zeros(img.shape)
for i in range(img.shape[2]):
    noisy_img[:,:,i] = img[:,:,i] + np.array(noise_factor*noise, dtype=np.int)

# convert data type for use
noisy_img = np.asarray(noisy_img, dtype=np.uint8)

##