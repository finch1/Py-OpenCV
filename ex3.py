# Date: 28/05/2018
from __future__ import print_function

from keras.datasets import mnist
from termcolor import colored

import matplotlib.pyplot as plt 

def show_shapes(x_train, y_train, x_test, y_test, color='green'):
    print(colored('Training shape:', color, attrs=['bold']))
    print('   x_train.shape:', x_train.shape)
    print('   y_train.shape:', y_train.shape)
    print(colored('\nTesting shape:', color, attrs=['bold']))
    print('   x_test.shape:', x_test.shape)
    print('   y_test.shape:', y_test.shape)

def show_sample(x_train, y_train, idx=0, color='blue'):

    for i in range(0,3):
        plt.subplot(1,3,i+1)
        plt.title("Label " + str(y_train[i]))
        img = x_train[i]
        img[img < 180] = 0
        plt.imshow(img, cmap='gray')
        # plt.axis('off')

    plt.show()


# download and load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# to know the size of data
print("Train data shape:", x_train.shape, "Test data shape:", x_test.shape)

show_shapes(x_train, y_train, x_test, y_test)
print('\n****************************\n')
show_sample(x_train, y_train)