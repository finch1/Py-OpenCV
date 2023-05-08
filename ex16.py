# Date: 06/06/2018

# CNN step by step
# set up convolution

from keras.layers import Conv2D, Input, Activation, MaxPooling2D
from keras.models import Model

def print_model():
    """
    Creates a sample model and prints output shape
    Use this to analyse convolution parameters. 
    """

    # create input with gicen shape
    x = Input(shape=(512, 512, 1))

    # create a convolution layer
    conv = Conv2D( filters = 32, 
                kernel_size = (5,5),
                strides = 1,
                padding = "same", # ="valid" lacks padding set, and the kernel cannot be applied to the edges of the input
                use_bias = True)(x)

    # add activation layer
    act = Activation('relu')(conv)

    # add max pooling layer
    pool = MaxPooling2D(pool_size = (2, 2))(act)

    # create model
    model = Model(inputs=x, outputs=pool)

    # prints our model created
    model.summary()

print_model()    

"""
What just happened:
    input of the shape 512x512x3
    convolution, we have 32 filters, 5x5 each
    strides set to 1
    padding for the edges, so kernel captures all the images
    no bias
    output after conv is of shape(None, 512, 512, 32) i.e. (samples, width, height, depth)
    total parameters for this layer: 5 x 5 x 3 x 32 = 2400 i.e. (kernel size * num of filters)
    
    On setting use_bias=True, it will add a constant value to each kernel
    and for a conv layer the bias parameter is the same as the number of filters used
    (5 x 5 x 3 x 32) + 32 = 2432
"""