from keras.initializers import *
import keras.backend as K
class Sigmoid_Init(Initializer):
    """Initializer that generates tensors uniform on log odds.
    """

    def __call__(self, shape, dtype=None):
        out = K.random_uniform(shape=shape,
                               minval=K.epsilon(),
                               maxval=1-K.epsilon(),
                               dtype=dtype)
        out = -K.log((1/out)-1)
        return out

class Softmax_Init(Initializer):
    """Initializer that generates tensors Uniform on simplex.
    """

    def __call__(self, shape, dtype=None):
        out = K.random_uniform(shape=shape,
                               minval=K.epsilon(),
                               maxval=1-K.epsilon(),
                               dtype=dtype)
        out = -K.log(out)
        out = K.log(out/K.sum(out, axis=2, keepdims=True))
        return out

class Exp_Init(Initializer):
    def __call__(self, shape, dtype=None):
        out = K.random_uniform(shape=shape,
                               minval=K.epsilon(),
                               maxval=1 - K.epsilon(),
                               dtype=dtype)
        out = -K.log(out)
        return out
