from keras.layers import Layer

from Activation.activations import *
from keras.layers import Dropout
import numpy as np
class Birelu(Layer):
	def __init__(self, activation,relu_birelu_sel=1,layer_index=0, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		if relu_birelu_sel==-1:
			self.decay_drop=True
		else:
			self.decay_drop=False
		if not relu_birelu_sel==1:
			self.uses_learning_phase=True
		super(Birelu, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		return [input_shape, input_shape]

	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		if self.decay_drop:
			a = .01*(2*self.layer_index+1)
			time_tensor= K.variable(value=0)
			time_update = K.update_add(time_tensor,0.002)
			dropout_rate = 1/(1+K.exp(-a*(time_update+3)))
			dropout_rate = (K.cos((a*time_update)+3)/10)+.8
		else:
			dropout_rate=self.relu_birelu_sel
		if self.activation.__name__ == 'relu':
			pas = relu(x)
			inv_pas = relu(-x)
		else:
			prob = self.activation(x)
			pas = x * prob
			inv_pas = x * (prob - 1)
		if not self.relu_birelu_sel==1:
			variable_placeholder =K.variable(0)
			one = K.ones_like(variable_placeholder)
			birelu_flag = K.random_binomial(K.shape(variable_placeholder),p=dropout_rate)
			pas_flag = K.random_binomial(K.shape(variable_placeholder),p=.5)
			inv_flag = one-pas_flag
			inv_flag = inv_flag+birelu_flag
			pas_flag = K.minimum(pas_flag+birelu_flag,one)
			inv_flag = K.minimum(inv_flag+birelu_flag,one)
			return [K.in_train_phase(pas*pas_flag,pas),K.in_train_phase(inv_pas*inv_flag,inv_pas)]
			# return [pas*pas_flag, inv_pas*inv_flag]
		return [pas,inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Relu(Layer):
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

	def __init__(self, activation, **kwargs):
		self.supports_masking = True
		self.activation = get(activation)
		super(Relu, self).__init__(**kwargs)

	def call(self, x, mask=None):
		if self.activation.__name__ == 'relu' or self.activation.__name__=='avr':
			pas = relu(x)
		else:
			prob = self.activation(x)
			pas = x * prob
		return pas

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Relu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
