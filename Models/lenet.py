# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Flatten, Dense, Input, Activation
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D
from keras import backend as K


def lenet_model(weights=None,
          input_tensor=(32,32,3),nb_classes=10):
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    # if K.image_dim_ordering() == 'th':
    #         input_shape = (3, 32, 32)
    # else:
    #         input_shape = (32, 32, 3)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor)
    #     else:
    #         img_input = input_tensor
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, input_shape=(input_tensor), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D())
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(AveragePooling2D())
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add((Flatten()))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # load weights
    return model


