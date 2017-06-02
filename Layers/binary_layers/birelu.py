from keras.layers import Layer

from Activation.activations import *
from keras.layers import Dropout
from keras import initializers
import tensorflow as tf
import numpy as np
class MaxEntropy(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = False
		super(MaxEntropy, self).__init__(**kwargs)
	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[2])
	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		# Input should be of size (batch,instance,class_nb)
		entropy = -K.sum(x*K.log(x),2)
		index = K.argmax(entropy,axis=1)

		K.gather()
		res = entropy[:,index,:]

		res = K.squeeze(res)
		return res
	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(MaxEntropy, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class Birelu(Layer):
	def __init__(self, activation,relu_birelu_sel=1,layer_index=0,leak_rate=0,child_p=.5,add_data=False, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		super(Birelu, self).__init__(**kwargs)
	def compute_output_shape(self, input_shape):
		return [input_shape, input_shape]
	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		if self.activation.__name__ == 'relu':
			pas = self.activation(x)
			inv_pas = self.activation(-x)
		else:
			assert 'activation is not relu'
		return [pas,inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class Crelu(Layer):
	def __init__(self, activation, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		super(Crelu, self).__init__(**kwargs)
	def compute_output_shape(self, input_shape):
		output_dim = 2*input_shape[1]
		output_shape = (input_shape[0],output_dim,input_shape[2],input_shape[3])
		return output_shape
	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		if self.activation.__name__ == 'relu':
			pas = self.activation(x)
			inv_pas = self.activation(-x)
			res=K.concatenate([pas,inv_pas],axis=1)
		else:
			raise ValueError('Activation is not Relu, For now Crelu only workds on relu')
		return res

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Crelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class Birelu_old(Layer):
	def __init__(self, activation,relu_birelu_sel=1,layer_index=0,leak_rate=0,child_p=.5,add_data=False, **kwargs):
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
		self.add_data = add_data
		self.leak_rate = leak_rate
		super(Birelu, self).__init__(**kwargs)
	def build(self, input_shape):
		self.concat_filter_num = int(self.leak_rate*input_shape[1])
	def get_output_shape_for(self, input_shape):
		output_filter_num = self.concat_filter_num+input_shape[1]
		output_shape = (input_shape[0],output_filter_num,input_shape[2],input_shape[3])
		return [output_shape, output_shape]
	def compute_output_shape(self, input_shape):
		output_filter_num = self.concat_filter_num + input_shape[1]
		output_shape = (input_shape[0], output_filter_num, input_shape[2], input_shape[3])
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
		if self.add_data:
			pas = pas+x
			inv_pas = inv_pas+x
		if not self.relu_birelu_sel==1:
			dropout_boostA = 1/(dropout_rate+((1-dropout_rate)*self.child_p))
			dropout_boostB = 1 / (dropout_rate + ((1 - dropout_rate) * (1-self.child_p)))
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
			pas = K.in_train_phase(dropout_boostA*pas*pas_flag,pas)
			inv_pas =K.in_train_phase(dropout_boostB*inv_pas*inv_flag,inv_pas)
			return [pas,inv_pas]
			return [pas*pas_flag, inv_pas*inv_flag]

		return [pas,inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class PermuteChannels(Layer):
	# Gets two tensors as input return 2^C output Tensors with selected channels from 2 tensors
	def __init__(self,max_perm,random_permute=False,**kwargs):
		self.supports_masking = False
		self.max_perm = max_perm
		self.output_number_of_tensor=max_perm
		self.random_permute = random_permute
		super(PermuteChannels, self).__init__(**kwargs)
	def build(self, input_shape):
		self.output_number_of_tensor= np.min((self.max_perm, 2 ** input_shape[0][1]))
		self.randomGen_op=[]
		if self.random_permute:
			for i in range(self.output_number_of_tensor):
				self.randomGen_op = self.randomGen_op+[K.random_binomial(shape=(1,input_shape[0][1],1,1),p=.8)]
		else:
			for i in range(self.output_number_of_tensor):
				self.randomGen_op +=[K.variable(np.random.randint(0, 2, (1, input_shape[0][1], 1, 1)))]
				# np.random.randint(0, 2, (1, input_shape[0][1], 1, 1)),
	def call(self, x, mask=None):
		pos = x[0]
		neg = x[1]
		permuted_tensor_list = []
		for i in range(self.output_number_of_tensor):
			index = self.randomGen_op[i]
			permuted_tensor_list+=[pos*index + neg*(1-index)]
		# output_tensor_filters = K.int_shape(pos)[1]
		# permuted_tensor_list = []
		# tensor_break_down = [[], []]
		# for i in range(output_tensor_filters):
		# 	tensor_break_down[0] = tensor_break_down[0] + [K.expand_dims(pos[:, i, :, :], 1)]
		# 	tensor_break_down[1] = tensor_break_down[1] + [K.expand_dims(neg[:, i, :, :], 1)]
		#
		# base_tensor = 0
		# tensor_index_list = np.random.randint(0,2**output_tensor_filters,self.output_number_of_tensor)
		# K.random_uniform(shape=)
		# for tensor_index in np.random.randint(0,2**output_tensor_filters):
		# 	perm_tensor = tensor_break_down[base_tensor]
		# 	base_tensor=1-base_tensor
		# 	binary_index = np.binary_repr(tensor_index, output_tensor_filters)
		# 	for channel_index in range(output_tensor_filters):
		# 		selector = int(binary_index[channel_index])
		# 		perm_tensor[channel_index] = tensor_break_down[selector][channel_index]
		# 	res_tensor = tf.concat(perm_tensor, 1)
		# 	permuted_tensor_list = permuted_tensor_list + [res_tensor]
		return permuted_tensor_list
	def compute_output_shape(self, input_shape):
		return self.output_number_of_tensor*[input_shape[0]]

	def get_output_shape_for(self, input_shape):
		shape = [input_shape]
		return input_shape
	def compute_mask(self, input, input_mask=None):
		return self.output_number_of_tensor*[None]
	def get_config(self):
		config = {'output_num_sensor': self.output_number_of_tensor}
		base_config = super(PermuteChannels, self).get_config()
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

class MaxoutDenseOverParallel(Layer):
	def __init__(self,**kwargs):
		self.supports_masking = False
		super(MaxoutDenseOverParallel, self).__init__(**kwargs)
	def call(self, x, mask=None):
		y = K.expand_dims(x,-1)
		y = K.squeeze(y,-1)
		y = K.permute_dimensions(y,[1,2,0])
		y = K.max(y,axis=-1)
		return y
	def get_output_shape_for(self, input_shape):
		return input_shape[0]
	def compute_mask(self, input, input_mask=None):
		return [None]

class InstanceDropout(Dropout):
    """Spatial 2D version of Dropout.

    This version performs the same function as Dropout, however it drops
    entire 2D feature maps instead of individual elements. If adjacent pixels
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout2D will help promote independence
    between feature maps and should be used instead.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        Same as input

    # References
        - [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)
    """

    def __init__(self, p, dim_ordering='default', **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        super(InstanceDropout, self).__init__(p, **kwargs)


	def _get_noise_shape(self, x):
		input_shape = K.shape(x)
		input_shape_list = K.int_shape(x)


		if self.dim_ordering == 'th':
			noise_shape = (input_shape[0], 1, 1, 1)
		elif self.dim_ordering == 'tf':
			noise_shape = (input_shape[0], 1, 1, 1)
		else:
			raise ValueError('Invalid dim_ordering:', self.dim_ordering)
		noise_shape = noise_shape[:input_shape_list.__len__()]
		return noise_shape
class FullyConnectedTensors(Layer):
	def __init__(self,output_tensors_len=0, init='one', weights=None, shared_axes=[1,2,3],
	             **kwargs):
		'if dropout dim is 0 then disabled if 1 : instances will be dropped'
		self.output_tensor_len = output_tensors_len
		self.supports_masking = True
		self.init = initializers.get(init)
		self.initial_weights = weights
		if not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)
		super(FullyConnectedTensors, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		res = []
		for i in range(self.output_tensor_len):
			res +=[input_shape[0]]
		return res
	def compute_mask(self, input, input_mask=None):
		res = []
		for i in range(self.output_tensor_len):
			res += [None]
		return res
	def build(self, input_shape):
		if self.output_tensor_len==0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len,input_shape.__len__()] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		#TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.eye(param_shape[0],param_shape[1])
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i+1] = 1
				self.param_broadcast[i+1] = True
		# self.alphas = K.zeros(param_shape)
		for i in range(param_shape.__len__()-2):
			weights= np.expand_dims(weights,-1)
		weights = weights*np.ones(param_shape)
		self.alphas = K.variable(weights,name='{}_alphas'.format(self.name))
		self.trainable_weights = [self.alphas]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights
	def call(self, x, mask=None):
		# if K.backend() == 'theano':
		# 	pos = (K.pattern_broadcast(self.alphas, self.param_broadcast) * pos)
		# else:
		# 	pos = self.alphas * pos
		result = []

		# y = K.expand_dims(x, 2)
		# y = K.permute_dimensions(y, [1, 2, 0, 3, 4, 5])
		# y = y*self.alphas
		# res = K.sum(y,2)
		# for i in range(self.output_tensor_len):
		# 	result+=[res[:,i,:,:,:]]
		for j in range(self.output_tensor_len):
			sum = K.zeros_like(x[0])
			for i in range(x.__len__()):
				sum+=x[i]*self.alphas[j,i,:,:,:]
			result+=[sum]

		return result

	def get_config(self):
		config = {}
		base_config = super(FullyConnectedTensors, self).get_config()
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

class DropInstanse(Layer):
	def __init__(self,dropInstance_rate,**kwargs):
		self.drop_rate = dropInstance_rate
		self.uses_learning_phase = True
		self.supports_masking = True
		super(DropInstanse,self).__init__(**kwargs)
		pass
	def call(self, x, mask=None):
		pass
	def get_output_shape_for(self, input_shape):
		pass
	def compute_mask(self, input, input_mask=None):
		tf.strided_slice()
		pass
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
		self.init = initializers.get(init)
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
		if self.activation.__name__ == 'relu':
			pas = relu(x)
		else:
			prob = self.activation(x)
			pas = x * prob
		return pas

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Relu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class AVR(Layer):
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

	def __init__(self,  **kwargs):
		self.supports_masking = True
		super(AVR, self).__init__(**kwargs)

	def call(self, x, mask=None):
		pas = K.abs(x)
		return pas

	def get_config(self):
		config = {'activation': 'avr'}
		base_config = super(AVR, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
