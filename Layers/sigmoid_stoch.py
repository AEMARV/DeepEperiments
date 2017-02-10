from keras.layers import Layer
from keras.layers.core import Activation
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

    def __init__(self,**kwargs):
        self.supports_masking = True
        self.activation = sigmoid_stoch
        super(StochSigmoidActivation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return self.activation(x)

    def get_config(self):
        config = {'activation': self.activation.__name__}
        base_config = super(StochSigmoidActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))