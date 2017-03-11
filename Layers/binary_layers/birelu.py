from keras.layers import Layer

from Activation.activations import *
from keras.layers import Dropout
from keras import initializations
import numpy as np
class PBirelu(Layer):
	"""Parametric Rectified Linear Unit.

	It follows:
	`f(x) = alphas * x for x < 0`,
	`f(x) = x for x >= 0`,
	where `alphas` is a learned array with the same shape as x.

	# Input shape
	    Arbitrary. Use the keyword argument `input_shape`
	    (tuple of integers, does not include the samples axis)
	    when using this layer as the first layer in a model.

	# Output shape
	    Same shape as the input.

	# Arguments
	    init: initialization function for the weights.
	    weights: initial weights, as a list of a single Numpy array.
	    shared_axes: the axes along which to share learnable
	        parameters for the activation function.
	        For example, if the incoming feature maps
	        are from a 2D convolution
	        with output shape `(batch, height, width, channels)`,
	        and you wish to share parameters across space
	        so that each filter only has one set of parameters,
	        set `shared_axes=[1, 2]`.

	# References
	    - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
	"""

	def __init__(self, init='zero', weights=None, shared_axes=None, **kwargs):
		self.supports_masking = True
		self.init = initializations.get(init)
		self.initial_weights = weights
		if not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)
		super(PBirelu, self).__init__(**kwargs)

	def build(self, input_shape):
		param_shape = list(input_shape[1:])
		self.param_broadcast = [False] * len(param_shape)
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i - 1] = 1
				self.param_broadcast[i - 1] = True

		self.alphas = self.init(param_shape, name='{}_alphas'.format(self.name))
		self.trainable_weights = [self.alphas]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def call(self, x, mask=None):
		pos = K.relu(x)
		if K.backend() == 'theano':
			neg = (K.pattern_broadcast(self.alphas, self.param_broadcast) * (x - K.abs(x)) * 0.5)
		else:
			neg = self.alphas * (x - K.abs(x)) * 0.5
		return pos + neg

	def get_config(self):
		config = {'init': self.init.__name__}
		base_config = super(PBirelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



class Duplicator(Layer):
	def __init__(self, relu_birelu_sel=1,layer_index=0,child_p=.5, **kwargs):
		self.supports_masking = False
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		self.child_p = child_p
		if not relu_birelu_sel==1:
			self.uses_learning_phase=True
		super(Duplicator, self).__init__(**kwargs)
	def get_output_shape_for(self, input_shape):
		return [input_shape, input_shape]

	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		dropout_rate=self.relu_birelu_sel
		pas = x
		inv_pas = x
		if not self.relu_birelu_sel==1:
			variable_placeholder =K.variable(0)
			one = K.ones_like(variable_placeholder)
			birelu_flag = K.random_binomial(K.shape(variable_placeholder),p=dropout_rate)
			pas_flag = K.random_binomial(K.shape(variable_placeholder),p=self.child_p)
			inv_flag = one-pas_flag
			inv_flag = inv_flag+birelu_flag
			pas_flag = K.minimum(pas_flag+birelu_flag,one)
			inv_flag = K.minimum(inv_flag+birelu_flag,one)
			pas = K.in_train_phase(pas*pas_flag,pas)
			inv_pas =K.in_train_phase(inv_pas*inv_flag,inv_pas)
			return [pas,inv_pas]
			# return [pas*pas_flag, inv_pas*inv_flag]

		return [pas,inv_pas]

	def get_config(self):
		config = {'child_bias_probability': str(self.child_p)}
		base_config = super(Duplicator, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class Slice(Layer):
	def __init__(self,nb_filter_to_slice, **kwargs):
		self.nb_filter = nb_filter_to_slice
		super(Slice, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		output_shape = (input_shape[0],self.nb_filter,input_shape[2],input_shape[3])
		return  output_shape
	def call(self, x, mask=None):
		return x[:,:self.nb_filter,:,:]
class Birelu(Layer):
	def __init__(self, activation,relu_birelu_sel=1,layer_index=0,leak_rate=0,child_p=.5, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		self.child_p = child_p
		if relu_birelu_sel==-1:
			self.decay_drop=True
		else:
			self.decay_drop=False
		if not relu_birelu_sel==1:
			self.uses_learning_phase=True
		self.leak_rate = leak_rate
		super(Birelu, self).__init__(**kwargs)
	def build(self, input_shape):
		self.concat_filter_num = int(self.leak_rate*input_shape[1])
	def get_output_shape_for(self, input_shape):
		output_filter_num = self.concat_filter_num+input_shape[1]
		output_shape = (input_shape[0],output_filter_num,input_shape[2],input_shape[3])
		return [output_shape, output_shape]

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
			pas_flag = K.random_binomial(K.shape(variable_placeholder),p=self.child_p)
			inv_flag = one-pas_flag
			inv_flag = inv_flag+birelu_flag
			pas_flag = K.minimum(pas_flag+birelu_flag,one)
			inv_flag = K.minimum(inv_flag+birelu_flag,one)
			pas_c = K.concatenate([pas, inv_pas[:,:self.concat_filter_num,:,:]],axis=1)
			inv_pas_c = K.concatenate([inv_pas,pas[:,:self.concat_filter_num,:,:]],axis=1)
			pas = pas_c
			inv_pas = inv_pas_c
			pas = K.in_train_phase(pas*pas_flag,pas)
			inv_pas =K.in_train_phase(inv_pas*inv_flag,inv_pas)
			return [pas,inv_pas]
			# return [pas*pas_flag, inv_pas*inv_flag]

		return [pas,inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class Slice(Layer):
	def __init__(self,nb_filter_to_slice, **kwargs):
		self.nb_filter = nb_filter_to_slice
		super(Slice, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		output_shape = (input_shape[0],self.nb_filter,input_shape[2],input_shape[3])
		return  output_shape
	def call(self, x, mask=None):
		return x[:,:self.nb_filter,:,:]
class Birelu_nary(Layer):
	def __init__(self, activation,nb_filter,relu_birelu_sel=1,layer_index=0, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		self.nb_filter = nb_filter
		if relu_birelu_sel==-1:
			self.decay_drop=True
		else:
			self.decay_drop=False
		if not relu_birelu_sel==1:
			self.uses_learning_phase=True
		super(Birelu_nary, self).__init__(**kwargs)
	def build(self, input_shape):
		self.W = self.add_weight((input_shape[1],1,1),trainable=True,initializer='uniform',
			name='{}_W'.format(
			self.name))
		# super(Birelu_nary, self).build(input_shape)
		self.built=True
	def get_output_shape_for(self, input_shape):
		res = []
		for i in range(2 ** input_shape[1]):
			res += [input_shape]
		return res

	def compute_mask(self, input, input_mask=None):
		res = []
		for i in range(2**self.nb_filter):
			res+=[None]
		return res

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
			# filter_prob = sigmoid(self.W)
			# # filter_prob = K.ones_like(x[:,:,0,0])*filter_prob
			# filter_set1 = filter_prob
			# # filter_set1 = K.random_binomial(K.shape(filter_prob),filter_prob)
			# filter_set2 = K.ones_like(filter_set1)-filter_set1
			# pas1 = (filter_set1*pas)+(filter_set2*inv_pas)
			# inv_pas1 = (filter_set2*pas)+(filter_set1*inv_pas)
			# pas = pas1
			# inv_pas = inv_pas1
		if not self.relu_birelu_sel==1:
			variable_placeholder =K.variable(0)
			one = K.ones_like(variable_placeholder)
			birelu_flag = K.random_binomial(K.shape(variable_placeholder),p=dropout_rate)
			pas_flag = K.random_binomial(K.shape(variable_placeholder),p=.5)
			inv_flag = one-pas_flag
			inv_flag = inv_flag+birelu_flag
			pas_flag = K.minimum(pas_flag+birelu_flag,one)
			inv_flag = K.minimum(inv_flag+birelu_flag,one)
			result = []
			pas = K.in_train_phase(pas * pas_flag, pas)
			inv_pas = K.in_train_phase(inv_pas * inv_flag, inv_pas)
			for i in range(2**self.nb_filter):
				filter_set1=[]
				for filter_index in range(self.nb_filter):
					filter_set1 +=[np.mod(i,2**(filter_index+1))]
				mask = K.variable(np.array(filter_set1))
				mask = K.expand_dims(K.expand_dims(K.expand_dims(mask,0),-1),-1)
				filter_set1 = mask
				filter_set2 = 1- mask
				pas1 = (filter_set1*pas)+(filter_set2*inv_pas)
				result+=[pas1]
			return result
			# return [pas*pas_flag, inv_pas*inv_flag]
		return [pas,inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu_nary, self).get_config()
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
