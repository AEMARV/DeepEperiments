import numpy as np
from keras import initializers
from keras.layers import Dropout
from keras.layers import Layer,InputSpec
from keras.utils import conv_utils
from utils.modelutils.activations.activations import *
from utils.modelutils.regularizer.entropy_activity_reg import SoftmaxEntropyRegularizer
from keras.legacy import interfaces

class _Pooling2D(Layer):
	"""Abstract class for different pooling 2D layers.
	"""

	def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
		super(_Pooling2D, self).__init__(**kwargs)
		data_format = conv_utils.normalize_data_format(data_format)
		if strides is None:
			strides = pool_size
		self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
		self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.input_spec = InputSpec(ndim=4)

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]
		rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding, self.strides[0])
		cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding, self.strides[1])
		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[3])

	def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
		raise NotImplementedError

	def call(self, inputs):
		output = self._pooling_function(inputs=inputs, pool_size=self.pool_size, strides=self.strides, padding=self.padding,
		                                data_format=self.data_format)
		return output

	def get_config(self):
		config = {
			'pool_size': self.pool_size, 'padding': self.padding, 'strides': self.strides, 'data_format': self.data_format
		}
		base_config = super(_Pooling2D, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Birelu(Layer):
	def __init__(self, activation, relu_birelu_sel=1, layer_index=0, leak_rate=0, child_p=.5, add_data=False, **kwargs):
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
		elif self.activation.__name__ == 'sigmoid':
			pas = self.activation(x)
			inv_pas = self.activation(-x)
		else:
			assert 'activation is not relu'
		return [pas, inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class MaskBirelu(Layer):
	def __init__(self, activation='relu', relu_birelu_sel=1, layer_index=0, leak_rate=0, child_p=.5, add_data=False, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		super(MaskBirelu, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return [input_shape, input_shape]

	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		pas = self.activation(x)
		inv_pas = self.activation(-x)
		return [pas, inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(MaskBirelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Grelu(Layer):
	def __init__(self, activation, relu_birelu_sel=1, layer_index=0, leak_rate=0, child_p=.5, add_data=False, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		super(Grelu, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return [input_shape, input_shape]

	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		if self.activation.__name__ == 'relu':
			pas = self.activation(x)
			inv_pas = self.activation(-x)
		elif self.activation.__name__ == 'sigmoid':
			pas = self.activation(x)
			inv_pas = self.activation(-x)
		else:
			assert 'activation is not relu'
		return [pas, inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Grelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class MaxEntropy(Layer):
	def __init__(self, **kwargs):
		self.tensor_list_len = 0
		super(MaxEntropy, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape[0]

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		# concat_x = K.stack(x, 2)
		# entropy = -K.sum(concat_x * K.log(concat_x + K.epsilon()), axis=1)
		# entropy = - entropy
		# entropy_zero_max =  entropy- K.max(entropy,axis=1,keepdims=True)
		# entropy = entropy_zero_max*1000
		# weights = softmax(entropy)
		# weights = K.expand_dims(weights, axis=1)
		# normalized_pred = weights * concat_x
		# pred = K.sum(normalized_pred, axis=2)

		concat_x = K.stack(x, 2)
		entropy = -K.sum(concat_x * K.log(concat_x + K.epsilon()), axis=1)
		min_entropy = K.repeat_elements(K.min(entropy, axis=1, keepdims=True), axis=1, rep=x.__len__())
		mask = K.equal(entropy, min_entropy)
		mask = K.cast(mask, K.floatx())
		weights = softmax(mask)
		mask = K.expand_dims(weights, 1)
		concat_x = mask * concat_x
		res = K.sum(concat_x, axis=2)

		# n = K.sum(mask, axis=1, keepdims=True)  # Making sure only one max exists ow. average between masks
		# weights = softmax(entropy)
		# mask = K.expand_dims(mask, 1)
		# concat_x = mask * concat_x
		# res = K.sum(concat_x, axis=2) / n

		return res


class WeightedAverageWithEntropy(Layer):
	# takes the weighted average of each image based on their entropy
	def __init__(self, **kwargs):
		self.tensor_list_len = 0
		super(WeightedAverageWithEntropy, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape[0]

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		concat_x = K.stack(x, 2)
		entropy = -K.sum(concat_x * K.log(concat_x + K.epsilon()), axis=1)
		entropy = K.max(entropy, axis=1, keepdims=True) - entropy
		weights = softmax(entropy)
		weights = K.expand_dims(weights, axis=1)
		normalized_pred = weights * concat_x
		pred = K.sum(normalized_pred, axis=2)

		return pred



class WeightedAverageWithEntropy0Max(Layer):
	# same as weighted average entropy but weights are based on -entropy instead of max(entropy)-entropy
	def __init__(self, **kwargs):
		self.tensor_list_len = 0
		super(WeightedAverageWithEntropy0Max, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape[0]

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		concat_x = K.stack(x, 2)
		entropy = -K.sum(concat_x * K.log(concat_x + K.epsilon()), axis=1)
		entropy = - entropy
		weights = softmax(entropy)
		weights = K.expand_dims(weights, axis=1)
		normalized_pred = weights * concat_x
		pred = K.sum(normalized_pred, axis=2)

		return pred


class RandomAveragePooling2D(_Pooling2D):

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]
		rows = conv_utils.conv_output_length(rows, self.pool_size[0], self.padding, self.strides[0])
		cols = conv_utils.conv_output_length(cols, self.pool_size[1], self.padding, self.strides[1])
		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[3])
	@interfaces.legacy_pooling2d_support
	def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs):
		super(RandomAveragePooling2D, self).__init__(pool_size, strides, padding, data_format, **kwargs)

	def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
		output_candidates = []
		output = K.pool2d(inputs, (1, 1), strides, padding, data_format, pool_mode='avg')
		for i in np.arange(np.log2(pool_size[0])):
			output += K.pool2d(inputs, (2**i,2**i), strides, padding, data_format, pool_mode='avg')
		# output = K.(output_candidates,axis=1)
		return output


class SoftmaxEntropyActivityRegLayer(Layer):
	# does nothing but adds the
	def __init__(self, **kwargs):
		self.activity = SoftmaxEntropyRegularizer()
		super(SoftmaxEntropyActivityRegLayer, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, inputs, mask=None):
		return len(inputs) * [None]

	def call(self, x, mask=None):
		self.add_loss([self.activity(x)], x)
		return x


class WeightedAverageSoftMax(Layer):
	# takes the weighted average of each image based on their entropy after softmax
	def __init__(self, **kwargs):
		self.tensor_list_len = 0
		super(WeightedAverageSoftMax, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape[0]

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		concat_x = K.stack(x, 2)
		concat_x_exp = K.exp(concat_x - K.max(concat_x, axis=1, keepdims=True))
		sum_expx = K.sum(concat_x_exp, axis=1, keepdims=True)
		softmax_concat_x = concat_x_exp / sum_expx
		entropy = -K.sum(softmax_concat_x * K.log(softmax_concat_x), axis=1)
		entropy = - entropy
		weights = softmax(entropy)
		weights = K.expand_dims(weights, axis=1)
		weighted_x = weights * concat_x
		weighted_average_x = K.sum(weighted_x, axis=2)
		pred = softmax(weighted_average_x)

		return pred


class WeightedAverageWithEntropy_Training(Layer):
	# takes the weighted average of each image based on their entropy
	def __init__(self, **kwargs):
		self.tensor_list_len = 0
		super(WeightedAverageWithEntropy, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape[0]

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		concat_x = K.stack(x, 2)
		entropy = -K.sum(concat_x * K.log(concat_x + K.epsilon()), axis=1)
		entropy = K.max(entropy, axis=1, keepdims=True) - entropy
		weights = softmax(entropy)
		weights = K.expand_dims(weights, axis=1)
		normalized_pred = weights * concat_x
		pred = K.sum(normalized_pred, axis=2)

		return pred


class Crelu(Layer):
	def __init__(self, activation, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		super(Crelu, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		output_dim = 2 * input_shape[1]
		output_shape = (input_shape[0], output_dim, input_shape[2], input_shape[3])
		return output_shape

	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		if self.activation.__name__ == 'relu':
			pas = self.activation(x)
			inv_pas = self.activation(-x)
			res = K.concatenate([pas, inv_pas], axis=1)
		else:
			raise ValueError('activations is not Relu, For now Crelu only workds on relu')
		return res

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Crelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Birelu_old(Layer):
	def __init__(self, activation, relu_birelu_sel=1, layer_index=0, leak_rate=0, child_p=.5, add_data=False, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		self.child_p = child_p
		if relu_birelu_sel == -1:
			self.decay_drop = True
		else:
			self.decay_drop = False
		if not relu_birelu_sel == 1:
			self.uses_learning_phase = True
		self.add_data = add_data
		self.leak_rate = leak_rate
		super(Birelu, self).__init__(**kwargs)

	def build(self, input_shape):
		self.concat_filter_num = int(self.leak_rate * input_shape[1])

	def get_output_shape_for(self, input_shape):
		output_filter_num = self.concat_filter_num + input_shape[1]
		output_shape = (input_shape[0], output_filter_num, input_shape[2], input_shape[3])
		return [output_shape, output_shape]

	def compute_output_shape(self, input_shape):
		output_filter_num = self.concat_filter_num + input_shape[1]
		output_shape = (input_shape[0], output_filter_num, input_shape[2], input_shape[3])
		return [output_shape, output_shape]

	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		if self.decay_drop:
			a = .01 * (2 * self.layer_index + 1)
			time_tensor = K.variable(value=0)
			time_update = K.update_add(time_tensor, 0.002)
			dropout_rate = 1 / (1 + K.exp(-a * (time_update + 3)))
			dropout_rate = (K.cos((a * time_update) + 3) / 10) + .8
		else:

			dropout_rate = self.relu_birelu_sel
		if self.activation.__name__ == 'relu':
			pas = relu(x)
			inv_pas = relu(-x)
		else:
			prob = self.activation(x)
			pas = x * prob
			inv_pas = x * (prob - 1)
		if self.add_data:
			pas = pas + x
			inv_pas = inv_pas + x
		if not self.relu_birelu_sel == 1:
			dropout_boostA = 1 / (dropout_rate + ((1 - dropout_rate) * self.child_p))
			dropout_boostB = 1 / (dropout_rate + ((1 - dropout_rate) * (1 - self.child_p)))
			variable_placeholder = K.variable(0)
			one = K.ones_like(variable_placeholder)
			birelu_flag = K.random_binomial(K.shape(variable_placeholder), p=dropout_rate)
			pas_flag = K.random_binomial(K.shape(variable_placeholder), p=self.child_p)
			inv_flag = one - pas_flag
			inv_flag = inv_flag + birelu_flag
			pas_flag = K.minimum(pas_flag + birelu_flag, one)
			inv_flag = K.minimum(inv_flag + birelu_flag, one)
			pas_c = K.concatenate([pas, inv_pas[:, :self.concat_filter_num, :, :]], axis=1)
			inv_pas_c = K.concatenate([inv_pas, pas[:, :self.concat_filter_num, :, :]], axis=1)
			pas = pas_c
			inv_pas = inv_pas_c
			pas = K.in_train_phase(dropout_boostA * pas * pas_flag, pas)
			inv_pas = K.in_train_phase(dropout_boostB * inv_pas * inv_flag, inv_pas)
			return [pas, inv_pas]
			return [pas * pas_flag, inv_pas * inv_flag]

		return [pas, inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class BiPermuteChannels(Layer):
	# Gets two tensors as input return 2^C output Tensors with selected channels from 2 tensors
	def __init__(self, max_perm, random_permute=False, p=.8, **kwargs):
		self.supports_masking = False
		self.max_perm = max_perm
		self.output_number_of_tensor = max_perm
		self.random_permute = random_permute
		self.p = p
		super(BiPermuteChannels, self).__init__(**kwargs)

	def build(self, input_shape):
		self.output_number_of_tensor = np.min((self.max_perm, 2 ** input_shape[0][1]))
		self.randomGen_op = []
		if self.random_permute:
			for i in range(self.output_number_of_tensor):
				self.randomGen_op = self.randomGen_op + [K.random_binomial(shape=(1, input_shape[0][1], 1, 1), p=self.p)]
		else:
			for i in range(self.output_number_of_tensor):
				self.randomGen_op += [K.variable(np.random.randint(0, 2, (1, input_shape[0][1], 1, 1)))]
			# np.random.randint(0, 2, (1, input_shape[0][1], 1, 1)),

	def call(self, x, mask=None):
		pos = x[0]
		neg = x[1]
		permuted_tensor_list = []
		for i in range(self.output_number_of_tensor):
			index = self.randomGen_op[i]
			permuted_tensor_list += [pos * index + neg * (1 - index), pos * (1 - index) + neg * (index)]
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
		return 2 * self.output_number_of_tensor * [input_shape[0]]

	def get_output_shape_for(self, input_shape):
		shape = [input_shape]
		return input_shape

	def compute_mask(self, input, input_mask=None):
		return 2 * self.output_number_of_tensor * [None]

	def get_config(self):
		config = {'output_num_sensor': self.output_number_of_tensor}
		base_config = super(BiPermuteChannels, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class MulNoise(Layer):
	# Gets two tensors as input return 2^C output Tensors with selected channels from 2 tensors
	def __init__(self, p=.5, shared_axes=[1, 2, 3], **kwargs):
		self.supports_masking = False
		self.p = p
		self.shared_axes = shared_axes
		self.randomGen_op = None
		super(MulNoise, self).__init__(**kwargs)

	def build(self, input_shape):
		param_shape = list(input_shape)
		for idx in self.shared_axes:
			param_shape[idx] = 1
		self.randomGen_op = K.random_binomial(shape=[1, param_shape[1], param_shape[2], param_shape[3]], p=self.p)

	def call(self, x, mask=None):
		noise = 2 * self.randomGen_op - 1
		res= K.in_train_phase(x*noise,x)
		return res

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, input, input_mask=None):
		return input_mask

	def get_config(self):
		config = {'p': self.p}
		base_config = super(MulNoise, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class PermuteChannels(Layer):
	# Gets two tensors as input return 2^C output Tensors with selected channels from 2 tensors
	def __init__(self, max_perm, random_permute=False, p=.8, **kwargs):
		self.supports_masking = False
		self.max_perm = max_perm
		self.output_number_of_tensor = max_perm
		self.random_permute = random_permute
		self.p = p
		super(PermuteChannels, self).__init__(**kwargs)

	def build(self, input_shape):
		self.output_number_of_tensor = np.min((self.max_perm, 2 ** input_shape[0][1]))
		self.randomGen_op = []
		if self.random_permute:
			for i in range(self.output_number_of_tensor):
				self.randomGen_op = self.randomGen_op + [K.random_binomial(shape=(1, input_shape[0][1], 1, 1), p=self.p)]
		else:
			for i in range(self.output_number_of_tensor):
				self.randomGen_op += [K.variable(np.random.randint(0, 2, (1, input_shape[0][1], 1, 1)))]
			# np.random.randint(0, 2, (1, input_shape[0][1], 1, 1)),

	def call(self, x, mask=None):
		pos = x[0]
		neg = x[1]
		permuted_tensor_list = []
		for i in range(self.output_number_of_tensor):
			index = self.randomGen_op[i]
			permuted_tensor_list += [pos * index + neg * (1 - index)]
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
		return self.output_number_of_tensor * [input_shape[0]]

	def get_output_shape_for(self, input_shape):
		shape = [input_shape]
		return input_shape

	def compute_mask(self, input, input_mask=None):
		return self.output_number_of_tensor * [None]

	def get_config(self):
		config = {'output_num_sensor': self.output_number_of_tensor}
		base_config = super(PermuteChannels, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Duplicator(Layer):
	def __init__(self, relu_birelu_sel=1, layer_index=0, child_p=.5, **kwargs):
		self.supports_masking = False
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		self.child_p = child_p
		if not relu_birelu_sel == 1:
			self.uses_learning_phase = True
		super(Duplicator, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		return [input_shape, input_shape]

	def compute_mask(self, input, input_mask=None):
		return [None, None]

	def call(self, x, mask=None):
		dropout_rate = self.relu_birelu_sel
		pas = x
		inv_pas = x
		if not self.relu_birelu_sel == 1:
			variable_placeholder = K.variable(0)
			one = K.ones_like(variable_placeholder)
			birelu_flag = K.random_binomial(K.shape(variable_placeholder), p=dropout_rate)
			pas_flag = K.random_binomial(K.shape(variable_placeholder), p=self.child_p)
			inv_flag = one - pas_flag
			inv_flag = inv_flag + birelu_flag
			pas_flag = K.minimum(pas_flag + birelu_flag, one)
			inv_flag = K.minimum(inv_flag + birelu_flag, one)
			pas = K.in_train_phase(pas * pas_flag, pas)
			inv_pas = K.in_train_phase(inv_pas * inv_flag, inv_pas)
			return [pas, inv_pas]
		# return [pas*pas_flag, inv_pas*inv_flag]

		return [pas, inv_pas]

	def get_config(self):
		config = {'child_bias_probability': str(self.child_p)}
		base_config = super(Duplicator, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Slice(Layer):
	def __init__(self, nb_filter_to_slice, **kwargs):
		self.nb_filter = nb_filter_to_slice
		super(Slice, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		output_shape = (input_shape[0], self.nb_filter, input_shape[2], input_shape[3])
		return output_shape

	def call(self, x, mask=None):
		return x[:, :self.nb_filter, :, :]


class MaxoutDenseOverParallel(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = False
		super(MaxoutDenseOverParallel, self).__init__(**kwargs)

	def call(self, x, mask=None):
		y = K.expand_dims(x, -1)
		y = K.squeeze(y, -1)
		y = K.permute_dimensions(y, [1, 2, 0])
		y = K.max(y, axis=-1)
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


class ConvBankAgg(Layer):
	def __init__(self,filter_size, init_val=0,conv_list_index_zero_not_shared=[],weights=None, shared_axes=[ 2, 3],
	             **kwargs):
		self.output_tensor_len = 1
		self.supports_masking = True
		# self.init = initializers.get(init)
		self.init_val = init_val
		self.initial_weights = weights
		if not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)
		self.conv_list=conv_list_index_zero_not_shared
		self.filter_size = filter_size
		super(ConvBankAgg, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return self.conv_list[0].compute_output_shape(input_shape)
	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return None

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len, len(self.conv_list)-1,self.filter_size,1,1]
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.ones((param_shape[0], param_shape[1]))
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i + 1] = 1
				self.param_broadcast[i + 1] = True
		# self.alphas = K.zeros(param_shape)
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)*self.init_val
		self.alphas = K.variable(weights, name='{}_alphas'.format(self.name))
		self.trainable_weights = [self.alphas]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def call(self, x, mask=None):
		weights = self.conv_list[0].weights[0]
		for idx,conv_layer in enumerate(self.conv_list[1:]):
			weights = weights*K.sigmoid(self.alphas[0, idx, :, :, :]) + (conv_layer.weights[0] * (1 - K.sigmoid(self.alphas[0, idx, :, :, :])))
		res = K.conv2d(x,weights,padding='same')

		return res
class TensorSelectSigmoid(Layer):
	def __init__(self, output_tensors_len=1, init='zero', weights=None, shared_axes=[1, 2, 3], **kwargs):
		self.output_tensor_len = output_tensors_len
		self.supports_masking = True
		self.init = initializers.get(init)
		self.initial_weights = weights
		if not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)
		super(TensorSelectSigmoid, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		res = []
		for i in range(self.output_tensor_len):
			res += [input_shape[0]]
		return res

	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return res

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len, 1] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.eye(param_shape[0], param_shape[1])
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i + 1] = 1
				self.param_broadcast[i + 1] = True
		# self.alphas = K.zeros(param_shape)
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)
		self.alphas = K.variable(weights, name='{}_alphas'.format(self.name))
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
			sum = (x[0] * K.sigmoid(self.alphas[j, 0, :, :, :]))+(x[1]* (1-K.sigmoid(self.alphas[j, 0, :, :, :])))
			result += [sum]

		return result


class FullyConnectedTensors(Layer):
	def __init__(self, output_tensors_len=0, init='one', weights=None, shared_axes=[1, 2, 3], **kwargs):
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
			res += [input_shape[0]]
		return res

	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return res

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.ones((param_shape[0]))
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i ] = 1
				self.param_broadcast[i ] = True
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)
		self.alphas = len(input_shape)*[None]
		for i in np.arange(len(input_shape)):
			if i==0:
				self.alphas[i] = K.variable(np.ones_like(weights), name='{}_alphas_branch_num{}'.format(self.name,i))
			else:
				self.alphas[i] = K.variable(np.zeros_like(weights), name='{}_alphas_branch_num{}'.format(self.name, i))
		self.trainable_weights = self.alphas

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def call(self, x, mask=None):
		result = []

		for j in range(self.output_tensor_len):
			sum = K.zeros_like(x[0])
			for i in range(x.__len__()):
				sum += x[i] * self.alphas[i][j, :, :, :]
			result += [sum]

		return result


class FullyConnectedTensorsv2(Layer):
	def __init__(self, output_tensors_len=0, init='one', weights=None, shared_axes=[1, 2, 3], **kwargs):
		self.output_tensor_len = output_tensors_len
		self.supports_masking = True
		self.init = initializers.get(init)
		self.initial_weights = weights
		if not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)
		super(FullyConnectedTensorsv2, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		res = []
		for i in range(self.output_tensor_len):
			res += [input_shape[0]]
		return res

	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return res

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.ones((param_shape[0]))
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i] = 1
				self.param_broadcast[i] = True
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)
		self.alphas = len(input_shape) * [None]
		for i in np.arange(len(input_shape)):
			self.alphas[i] = K.variable(np.ones_like(weights), name='{}_alphas_branch_num{}'.format(self.name, i))
		self.trainable_weights = self.alphas

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def call(self, x, mask=None):
		sum_weights = []
		for j in range(self.output_tensor_len):
			sum = K.zeros_like(self.alphas[0][0,:,:,:])
			for i in range(x.__len__()):
				sum += K.abs(K.mean(self.alphas[i][j, :, :, :],axis=[1,2],keepdims=True))

			sum_weights += [sum]
		result=[]
		for j in range(self.output_tensor_len):
			sum = K.zeros_like(x[0])
			for i in range(x.__len__()):
				sum += x[i] * (self.alphas[i][j, :, :, :]/sum_weights[j])

			result += [sum]

		return result

class FullyConnectedTensors_old(Layer):
	def __init__(self, output_tensors_len=0, init='one', weights=None, shared_axes=[1, 2, 3], **kwargs):
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
			res += [input_shape[0]]
		return res

	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return res

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len, input_shape.__len__()] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.eye(param_shape[0], param_shape[1])
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i + 1] = 1
				self.param_broadcast[i + 1] = True
		# self.alphas = K.zeros(param_shape)
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)
		self.alphas = K.variable(weights, name='{}_alphas_histshow'.format(self.name))
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
				sum += x[i] * self.alphas[j, i, :, :, :]
			result += [sum]

		return result


class FullyConnectedTensorsTanh(Layer):
	def __init__(self, output_tensors_len=0, init='one', weights=None, shared_axes=[1, 2, 3], **kwargs):
		self.output_tensor_len = output_tensors_len
		self.supports_masking = True
		self.init = initializers.get(init)
		self.initial_weights = weights
		if not isinstance(shared_axes, (list, tuple)):
			self.shared_axes = [shared_axes]
		else:
			self.shared_axes = list(shared_axes)
		super(FullyConnectedTensorsTanh, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		res = []
		for i in range(self.output_tensor_len):
			res += [input_shape[0]]
		return res

	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return res

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len, input_shape.__len__()] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.eye(param_shape[0], param_shape[1])
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i + 1] = 1
				self.param_broadcast[i + 1] = True
		# self.alphas = K.zeros(param_shape)
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)
		self.alphas = K.variable(weights, name='{}_alphas'.format(self.name))
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
				sum += x[i] * tanh(self.alphas[j, i, :, :, :])
			result += [sum]

		return result

class FullyConnectedTensorsLegacy(Layer):
	def __init__(self, output_tensors_len=0, init='one', weights=None, shared_axes=[1, 2, 3], **kwargs):
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
			res += [input_shape[0]]
		return res

	def compute_mask(self, input, input_mask=None):
		res = self.output_tensor_len * [None]
		return res

	def build(self, input_shape):
		if self.output_tensor_len == 0:
			self.output_tensor_len = input_shape.__len__()
		param_shape = [self.output_tensor_len, input_shape.__len__()] + list(input_shape[0][1:])
		self.param_broadcast = [False] * len(param_shape)
		# TODO Alpha is only compatible for square fully connected e.g [4x4]
		weights = np.eye(param_shape[0], param_shape[1])
		if self.shared_axes[0] is not None:
			for i in self.shared_axes:
				param_shape[i + 1] = 1
				self.param_broadcast[i + 1] = True
		# self.alphas = K.zeros(param_shape)
		for i in range(param_shape.__len__() - 2):
			weights = np.expand_dims(weights, -1)
		weights = weights * np.ones(param_shape)
		self.alphas = K.variable(weights, name='{}_alphas'.format(self.name))
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
				sum += x[i] * self.alphas[j, i, :, :, :]
			result += [sum]

		return result

	def get_config(self):
		config = {}
		base_config = super(FullyConnectedTensors, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Slice(Layer):
	def __init__(self, nb_filter_to_slice, **kwargs):
		self.nb_filter = nb_filter_to_slice
		super(Slice, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		output_shape = (input_shape[0], self.nb_filter, input_shape[2], input_shape[3])
		return output_shape

	def call(self, x, mask=None):
		return x[:, :self.nb_filter, :, :]


class Split(Layer):
	def __init__(self, nb_filter_to_slice, **kwargs):
		self.nb_filter = nb_filter_to_slice
		super(Split, self).__init__(**kwargs)

	def get_output_shape_for(self, input_shape):
		output_shape = (input_shape[0], self.nb_filter, input_shape[2], input_shape[3])
		output_shape_1 = (input_shape[0], input_shape[1]-self.nb_filter, input_shape[2], input_shape[3])
		return [output_shape,output_shape_1]
	def compute_output_shape(self, input_shape):
		output_shape = (input_shape[0], self.nb_filter, input_shape[2], input_shape[3])
		output_shape_1 = (input_shape[0], input_shape[1] - self.nb_filter, input_shape[2], input_shape[3])
		return [output_shape, output_shape_1]
	def compute_mask(self, inputs, mask=None):
		return [None,None]
	def call(self, x, mask=None):
		return [x[:, :self.nb_filter, :, :], x[:, self.nb_filter:, :, :]]


class Birelu_nary(Layer):
	def __init__(self, activation, nb_filter, relu_birelu_sel=1, layer_index=0, **kwargs):
		self.supports_masking = False
		self.activation = get(activation)
		self.relu_birelu_sel = relu_birelu_sel
		self.layer_index = layer_index
		self.nb_filter = nb_filter
		if relu_birelu_sel == -1:
			self.decay_drop = True
		else:
			self.decay_drop = False
		if not relu_birelu_sel == 1:
			self.uses_learning_phase = True
		super(Birelu_nary, self).__init__(**kwargs)

	def build(self, input_shape):
		self.W = self.add_weight((input_shape[1], 1, 1), trainable=True, initializer='uniform', name='{}_W'.format(self.name))
		# super(Birelu_nary, self).build(input_shape)
		self.built = True

	def get_output_shape_for(self, input_shape):
		res = []
		for i in range(2 ** input_shape[1]):
			res += [input_shape]
		return res

	def compute_mask(self, input, input_mask=None):
		res = []
		for i in range(2 ** self.nb_filter):
			res += [None]
		return res

	def call(self, x, mask=None):
		if self.decay_drop:
			a = .01 * (2 * self.layer_index + 1)
			time_tensor = K.variable(value=0)
			time_update = K.update_add(time_tensor, 0.002)
			dropout_rate = 1 / (1 + K.exp(-a * (time_update + 3)))
			dropout_rate = (K.cos((a * time_update) + 3) / 10) + .8
		else:
			dropout_rate = self.relu_birelu_sel
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
		if not self.relu_birelu_sel == 1:
			variable_placeholder = K.variable(0)
			one = K.ones_like(variable_placeholder)
			birelu_flag = K.random_binomial(K.shape(variable_placeholder), p=dropout_rate)
			pas_flag = K.random_binomial(K.shape(variable_placeholder), p=.5)
			inv_flag = one - pas_flag
			inv_flag = inv_flag + birelu_flag
			pas_flag = K.minimum(pas_flag + birelu_flag, one)
			inv_flag = K.minimum(inv_flag + birelu_flag, one)
			result = []
			pas = K.in_train_phase(pas * pas_flag, pas)
			inv_pas = K.in_train_phase(inv_pas * inv_flag, inv_pas)
			for i in range(2 ** self.nb_filter):
				filter_set1 = []
				for filter_index in range(self.nb_filter):
					filter_set1 += [np.mod(i, 2 ** (filter_index + 1))]
				mask = K.variable(np.array(filter_set1))
				mask = K.expand_dims(K.expand_dims(K.expand_dims(mask, 0), -1), -1)
				filter_set1 = mask
				filter_set2 = 1 - mask
				pas1 = (filter_set1 * pas) + (filter_set2 * inv_pas)
				result += [pas1]
			return result
		# return [pas*pas_flag, inv_pas*inv_flag]
		return [pas, inv_pas]

	def get_config(self):
		config = {'activation': self.activation.__name__}
		base_config = super(Birelu_nary, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


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

	def __init__(self, **kwargs):
		self.supports_masking = True
		super(AVR, self).__init__(**kwargs)

	def call(self, x, mask=None):
		pas = K.abs(x)
		return pas

	def get_config(self):
		config = {'activation': 'avr'}
		base_config = super(AVR, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
