from keras import backend as K
from keras.layers import Convolution2D
from keras.layers.core import Lambda
from keras.layers import merge
from keras.engine.topology import Layer
from theano import tensor
from keras import initializations
import numpy as np

def gate_layer(input_tensor, nb_filter, filter_size,input_shape = (None,None,None)):
    ''' Layer used to gate the convolution output. it will gate the output of a convolutional layer'''
    gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation='sigmoid',
                                input_shape=input_shape, border_mode='same')(input_tensor)
    data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation='relu', input_shape=input_shape,
                              border_mode='same')(input_tensor)
    merged = merge([data_conv, gate_output], mode='mul')
    return merged
def gated_layers_sequence(input_tensor,total_layers , nb_filter,filter_size,input_shape = (None,None,None)):
    out = input_tensor
    for i in np.arange(0,total_layers):
        out = gate_layer(out,nb_filter,filter_size,input_shape)
    return out
