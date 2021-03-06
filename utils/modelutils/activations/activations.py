from keras.activations import sigmoid
import keras.backend as K
import tensorflow as tf
# from keras.utils.generic_utils import get_from_module
from keras.layers import SpatialDropout2D
import six
from keras.utils.generic_utils import serialize_keras_object,deserialize_keras_object
def softplus_stoch(x):
    y = softplus(x)
    shape_x = K.shape(y)
    active_bool = K.lesser_equal(K.random_uniform(shape_x),y)
    res = tf.where(active_bool, K.ones_like(y), K.zeros_like(y))
    return res
def sigmoid_stoch(x):
    y = sigmoid(x)
    shape_x = K.shape(y)
    active_bool = K.lesser_equal(K.random_uniform(shape_x),y)
    res = tf.where(active_bool, K.ones_like(y), K.zeros_like(y))
    return res
def stoch_activation_function(x):
    shape_x = K.shape(x)
    active_bool = K.lesser_equal(K.random_uniform(shape_x),x)
    res = tf.where(active_bool, K.ones_like(x), K.zeros_like(x))
    return res
def softmax(x):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim == 3:
        e = K.exp(x - K.max(x, axis=-1, keepdims=True))
        s = K.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor '
                         'that is not 2D or 3D. '
                         'Here, ndim=' + str(ndim))
def neutral(x):
    return x
def negative(x):
    return K.zeros_like(x)-x
def avr(x):
    return relu(x,alpha=-1)
def inverter(x):
    return K.ones_like(x)-x
def elu(x, alpha=1.0):
    return K.elu(x, alpha)
def xlog(x):
    return K.relu(x)*K.log(K.abs(x)+K.epsilon())

def softplus(x):
    return K.softplus(x)


def softsign(x):
    return K.softsign(x)

def relu(x, alpha=0., max_value=None):
    return K.relu(x, alpha=alpha, max_value=max_value)


def tanh(x):
    return K.tanh(x)


def sigmoid(x):
    return K.sigmoid(x)


def hard_sigmoid(x):
    return K.hard_sigmoid(x)


def linear(x):
    return x

def serialize(initializer):
    return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='initializer')


def get(identifier):
    return globals()[identifier]

