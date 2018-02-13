from keras.layers import Layer
import keras as k
import tensorflow as tf


class LogSoftmax(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = False
        super(LogSoftmax, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def call(self, x, mask=None):
        res = []
        for xs in x:
            res += [tf.nn.log_softmax(xs, dim=2)]
        return res

    def get_config(self):
        base_config = super(LogSoftmax, self).get_config()
        return dict(list(base_config.items()))


class KlConv2D(k.layers.Conv2D):

    def __init__(self, **kwargs):
        self.supports_masking = False
        super(KlConv2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return [input_shape, input_shape]
    def build(self, input_shape):
        self.kernel = self.add_weight(name='weight',
                                      shape = [self.input_shape,self.output_shape],
                                      dtype ='double',
                                      initializer=k.initializers.)

    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def call(self, x, mask=None):
        res = []
        self.set_weights(tf.nn.softmax(self.kernel, dim=1))
        super(KlConv2D, self).call(x, mask)
        return res
    def get_config(self):
        base_config = super(KlConv2D, self).get_config()
        return dict(list(base_config.items()))

