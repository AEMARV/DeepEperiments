# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

from keras.layers import Flatten, Dense, Input, Activation,Dropout
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.regularizers import l1,l2,l1_l2
from keras.engine import Model
from utils.opt_utils import get_data_activation,get_filter_size
import numpy as np
def lenet_amir_model(opts,weights=None,
          input_tensor=None,nb_classes=10,input_shape=(3,32,32)):
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
    channel_expand_ratio = opts['model_opts']['param_dict']['param_expand']['rate']
    w_regularizer_str = opts['model_opts']['param_dict']['w_regularizer']['method']
    if w_regularizer_str == 'l1':
        w_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
    if w_regularizer_str==None:
        w_reg = None
    if w_regularizer_str == 'l2':
        w_reg = l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
    if input_tensor is None:
        img_input = Input(shape=input_shape, name='image_batch')
    else:
        img_input = input_tensor
    filter_size = get_filter_size(opts)
    if filter_size==-1:
        f_size = [5,3,5,4]
    else:
        f_size = np.min([[32,16,7,3],[filter_size,filter_size,filter_size,filter_size]],0)
    data_activation = get_data_activation(opts)
	#							Layer 1 Conv 5x5 32ch  border 'same' Max Pooling 3x3 stride 2 border valid
    x =Convolution2D(int(32*channel_expand_ratio), f_size[0], f_size[0], border_mode='same',
                            input_shape=(3,32,32),W_regularizer=w_reg,activation=None)(img_input)
    x = Activation(data_activation)(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same')(x)


	#                           Layer 2 Conv 3x3 64ch  border 'same' AveragePooling 3x3 stride 2
    x=Convolution2D(int(64 * channel_expand_ratio), f_size[1], f_size[1], W_regularizer=w_reg,activation=None,
                            border_mode='same')(x)
    x = Activation(data_activation)(x)
    x=AveragePooling2D(pool_size=(3, 3),strides=(2,2),border_mode='same')(x)

	#                           Layer 3 Conv 5x5 128ch  border 'same' AveragePooling3x3 stride 2
    x=Convolution2D(int(128*channel_expand_ratio), f_size[2], f_size[2], border_mode='same',W_regularizer=w_reg,
                    activation=None)(x)
    x=Activation(data_activation)(x)
    x=AveragePooling2D(pool_size=(3,3),strides=(2,2),border_mode='same')(x)

    #                           Layer 4 Conv 4x4 64ch  border 'same' no pooling
    x=Convolution2D(int(64*channel_expand_ratio), f_size[3], f_size[3],W_regularizer=w_reg,activation=None)(x)
    x=Activation(data_activation)(x)


    #                            Layer 5 Softmax
    x=Flatten()(x)
    x=Dense(nb_classes)(x)
    x=Activation('softmax')(x)
    model= Model(img_input,x)
    return model


