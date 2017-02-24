from keras.layers import Layer
from keras.layers.core import Activation
import keras.backend as K
from keras.layers import merge
from Activation.activations import stoch_activation_function,negative,inverter
class StochActivation(Layer):
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

    def __init__(self,**kwargs):
        self.supports_masking = True
        self.activation = stoch_activation_function
        super(StochActivation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        res = self.activation(x)
        return res

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(StochActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Negater(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.activation = negative
        super(Negater, self).__init__(**kwargs)

    def call(self, x, mask=None):
        res = self.activation(x)
        return res

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(Negater, self).get_config()
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
        self.activation = inverter
        super(Inverter, self).__init__(**kwargs)

    def call(self, x, mask=None):
        res = self.activation(x)
        return res

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(Inverter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

