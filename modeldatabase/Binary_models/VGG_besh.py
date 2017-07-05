# -*- coding: utf-8 -*-
"""VGG16 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

"""
from __future__ import absolute_import
from __future__ import print_function

import warnings

from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers.merge import add
from keras.models import Model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file

from modeldatabase.Binary_models.model_constructor_utils import node_list_to_list
from utils.modelutils.layers.binary_layers import Birelu

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def Layer_on_list(layer, tensor_list):
	res = []
	tensor_list = node_list_to_list(tensor_list)
	for x in tensor_list:
		res+=[layer(x)]
	return res
def Relu_birelu(x,sel='r'):
	if sel=='r':
		res = Layer_on_list(Activation('relu'),x)
	else:
		res = Layer_on_list(Birelu('relu'),x)
	return res
def besh_vgg1(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000):

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = [Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv1',input_shape=input_shape)(img_input)]
    x= Relu_birelu(x, sel='b')
    x = Layer_on_list(Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv2'), x)
    x=Relu_birelu(x, sel='r')
    x = Layer_on_list(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),x)

    # Block 2
    x = Layer_on_list(Conv2D(128, (3, 3), activation=None, padding='same', name='block2_conv1'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(128, (3, 3), activation=None, padding='same', name='block2_conv2'), x)
    x =Relu_birelu(x, sel='r')
    x = Layer_on_list(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),x)

    # Block 3
    x =Layer_on_list(Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv1'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv2'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv3'), x)
    x =Relu_birelu(x, sel='r')
    x = Layer_on_list(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),x)

    # Block 4
    x =Layer_on_list(Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv1'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv2'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv3'), x)
    x =Relu_birelu(x, sel='r')
    x = Layer_on_list(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),x)

    # Block 5
    x =Layer_on_list(Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv1'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv2'), x)
    x =Relu_birelu(x, sel='r')
    x =Layer_on_list(Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv3'), x)
    x =Relu_birelu(x, sel='r')
    x = Layer_on_list(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'),x)

    if include_top:
        # Classification block
        x = Layer_on_list(Flatten(name='flatten'),x)
        x = Layer_on_list(Dense(4096, activation='relu', name='fc1'),x)
        x = Layer_on_list(Dense(4096, activation='relu', name='fc2'),x)
        x = Layer_on_list(Dense(classes, activation=None, name='predictions'),x)
        x = add(x)
        x = Activation('softmax')(x)
    else:
		x = add(x)
		if pooling == 'avg':
		    x = GlobalAveragePooling2D()(x)
		elif pooling == 'max':
		    x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='vgg16')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model
if __name__ == '__main__':
    besh_vgg1(include_top=False,classes=10)