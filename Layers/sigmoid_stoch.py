from keras.layers import Layer
from keras.layers.core import Activation
import keras.backend as K
from keras.layers import merge
from Activation.activations import sigmoid_stoch
class StochSigmoidActivation(Layer):
    """Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self,opts,tan,**kwargs):
        self.supports_masking = True
        self.activation = sigmoid_stoch
        self.opts = opts
        self.tan=tan
        super(StochSigmoidActivation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.tan:
            res = self.activation(x, self.opts, self.tan)
            # res = self.activation(y, self.opts, self.tan)
        else:
            res = self.activation(x, self.opts, self.tan)
        ## adding concat
        # one = K.ones_like(res)
        # inverted = one-res
        # res = merge([res,inverted],mode='concat',concat_axis=1)
        return res

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(StochSigmoidActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Inverter(Layer):
    """Applies an activation function to an output.

    # Arguments
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a Theano or TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        self.activation = sigmoid_stoch
        super(Inverter, self).__init__(**kwargs)

    def call(self, x, mask=None):
        one = K.ones_like(x)
        res = one-x
        return res

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(StochSigmoidActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

