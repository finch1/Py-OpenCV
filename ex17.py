# Date: 07/06/2018
# CNN in practice

import keras
import keras.backend as K 

from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Input, Activation, MaxPooling2D, Dense, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

"""
we define the input height and width parameters to be used throughout, as
well as other parameters. Here, an epoch defines one iteration over all of
the data. So, the number of epochs means the total number of iterations over
all of the data"""

# setup parameters
batch_sz = 128      # batch size
nb_class = 10       # target number of classes
nb_epochs = 10      # training epochs
img_h, img_w = 28, 28   # input dims

""" 
download and prepare the dataset for training and validation"""

def get_dataset():
    """
    Return processed and reshaped dataset for training.
    In this case, Fashion-mnist dataset"""
    
    # so the x is the input and y is the label
    (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

    # test and train datasets
    print("Nb Train: ", x_train.shape[0], "Nb test: ", x_test.shape[0])
    x_train = x_train.reshape(x_train.shape[0], img_h, img_w, 1)
    x_test = x_test.reshape(x_test.shape[0], img_h, img_w, 1)
    in_shape = (img_h, img_w, 1)

    # normalize inputs
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0

    # convert to one hot vectors
    y_train = keras.utils.to_categorical(y_train, nb_class)
    y_test = keras.utils.to_categorical(y_test, nb_class)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = get_dataset()


"""
wrapper around convolution layer
Inputs:
    input_x: input layer / tensor - the image
    nb_filter: Number of filters for conv
"""
def conv3x3(input_x, nb_filters):
    # could have also done
    """
    # create a convolution layer
    conv = Conv2D(  filter = nb_filters,
                    kernel_size = (3,3),
                    strides = 1,
                    padding = "same",
                    use_bias = False)(input_x)
    
    # add activation
    act = Activation('relu')(conv)

    return act
    """

    return Conv2D(  nb_filters, kernel_size = (3,3), use_bias = False, 
                    activation = ('relu'),padding = "same")(input_x)    


"""
Creates a CNN model for training.
Inputs:
    img_h: input image height
    img_w: input image width
Returns:
    Model structure
"""

def create_model(img_h = 28, img_w = 28):
    inputs = Input(shape=(img_h, img_w, 1))

    x = conv3x3(inputs, 32)
    x = conv3x3(x, 32)
    # max pooling layer
    x = MaxPooling2D(pool_size = (2,2))(x)   
    x = conv3x3(x, 64)
    x = conv3x3(x, 64)
    x = MaxPooling2D(pool_size = (2,2))(x) 
    x = conv3x3(x, 128)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation = "relu")(x)
    # overall layer
    preds = Dense(nb_class, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = preds)
    print(model.summary())
    return model

model = create_model()

# setup the optimizer, loss function and metrics for model
model.compile(  loss = keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adam(),
                metrics=['accuracy'])

# Optional: To save model after each epoch of training
callback = ModelCheckpoint('mnist_cnn.h5')                

# Train the model
model.fit(  x_train, y_train,
            batch_size= batch_sz,
            epochs= nb_epochs,
            verbose= 1,
            validation_data= (x_test, y_test),
            callbacks= [callback])

# evaluate and print accuracy
score = model.evaluate(x_test, y_test, verbose= 0)
print('Test loss: ', score[0])
print('Test accuarcy: ', score[1])
