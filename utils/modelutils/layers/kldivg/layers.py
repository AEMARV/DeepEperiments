from keras.layers import Layer
import keras as k
import tensorflow as tf
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.layers.pooling import AveragePooling2D
from keras.legacy import interfaces


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

	def __init__(self, rank,
	             filters,
	             kernel_size,
	             strides=1,
	             padding='valid',
	             data_format=None,
	             dilation_rate=1,
	             activation=None,
	             use_bias=False,
	             kernel_initializer='glorot_uniform',
	             bias_initializer='zeros',
	             kernel_regularizer=None,
	             bias_regularizer=None,
	             activity_regularizer=None,
	             kernel_constraint=None,
	             bias_constraint=None,
	             **kwargs):
		super(KlConv2D, self).__init__(**kwargs)
		self.rank = rank
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
		self.activation = activations.get(activation)
		self.use_bias = False
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = None
		self.bias_regularizer = None
		self.activity_regularizer = None
		self.kernel_constraint = None
		self.bias_constraint = None
		self.input_spec = InputSpec(ndim=self.rank + 2)

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(
					space[i],
					self.kernel_size[i],
					padding=self.padding,
					stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(
					space[i],
					self.kernel_size[i],
					padding=self.padding,
					stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], self.filters) + tuple(new_space)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
			                 'should be defined. Found `None`.')
		input_dim = input_shape[channel_axis]
		kernel_shape = self.kernel_size + (input_dim, self.filters)

		self.kernel = self.add_weight(shape=kernel_shape,
		                              initializer=self.kernel_initializer,
		                              name='kernel',
		                              regularizer=self.kernel_regularizer,
		                              constraint=self.kernel_constraint)
		self.const_kernel = self.add_weight(shape=kernel_shape,
		                                    initializer=k.initializers.ones,
		                                    name='kernel',
		                                    regularizer=self.kernel_regularizer,
		                                    constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,),
			                            initializer=self.bias_initializer,
			                            name='bias',
			                            regularizer=self.bias_regularizer,
			                            constraint=self.bias_constraint)
		else:
			self.bias = None
		# Set input spec.
		self.input_spec = k.engine.InputSpec(ndim=self.rank + 2,
		                                     axes={channel_axis: input_dim})

		self.normalize_weights()
		self.built = True

	def compute_mask(self, input, input_mask=None):
		return None

	def normalize_weights(self):
		self.kernel = self.kernel - k.backend.logsumexp(self.kernel,
		                                                axis=1,
		                                                keepdims=False)

	def ent_kernel(self):
		e = -self.kernel * k.backend.exp(self.kernel)
		e = k.backend.sum(e,1,keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		e = k.backend.sum(e, 3, keepdims=True)
		return e

	def call(self, x, mask=None):
		self.normalize_weights()
		xprob = k.backend.exp(x)
		cross = k.backend.conv2d(xprob,
		                         self.kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)

		ent_ker = self.ent_kernel()
		ent_x = k.backend.conv2d(xprob*x,
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)

		return cross - ent_ker + ent_x

	def get_config(self):
		base_config = super(KlConv2D, self).get_config()
		return base_config


class KlConv2Db(k.layers.Conv2D):
	def __init__(self, rank,
	             filters,
	             kernel_size,
	             strides=1,
	             padding='valid',
	             data_format=None,
	             dilation_rate=1,
	             activation=None,
	             use_bias=False,
	             kernel_initializer='glorot_uniform',
	             bias_initializer='zeros',
	             kernel_regularizer=None,
	             bias_regularizer=None,
	             activity_regularizer=None,
	             kernel_constraint=None,
	             bias_constraint=None,
	             **kwargs):
		super(KlConv2Db, self).__init__(**kwargs)
		self.rank = rank
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
		self.activation = activations.get(activation)
		self.use_bias = False
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = None
		self.bias_regularizer = None
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(ndim=self.rank + 2)

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(
					space[i],
					self.kernel_size[i],
					padding=self.padding,
					stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(
					space[i],
					self.kernel_size[i],
					padding=self.padding,
					stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], self.filters) + tuple(new_space)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
			                 'should be defined. Found `None`.')
		input_dim = input_shape[channel_axis]
		kernel_shape = self.kernel_size + (input_dim, self.filters)

		self.kernel = self.add_weight(shape=kernel_shape,
		                              initializer=self.kernel_initializer,
		                              name='kernel',
		                              regularizer=self.kernel_regularizer,
		                              constraint=self.kernel_constraint)
		self.const_kernel = self.add_weight(shape=kernel_shape,
		                                    initializer=k.initializers.ones,
		                                    name='kernel',
		                                    regularizer=self.kernel_regularizer,
		                                    constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,),
			                            initializer=self.bias_initializer,
			                            name='bias',
			                            regularizer=self.bias_regularizer,
			                            constraint=self.bias_constraint)
		else:
			self.bias = None
		# Set input spec.
		self.input_spec = k.engine.InputSpec(ndim=self.rank + 2,
		                                     axes={channel_axis: input_dim})

		self.normalize_weights()
		self.built = True

	def compute_mask(self, input, input_mask=None):
		return None

	def normalize_weights(self):
		self.kernel = self.kernel - k.backend.logsumexp(self.kernel,
		                                                axis=1,
		                                                keepdims=False)

	def ent_kernel(self):
		e1 = k.backend.sigmoid(self.kernel)*k.backend.softplus(self.kernel)
		e0 = k.backend.sigmoid(-self.kernel)*k.backend.softplus(-self.kernel)
		e = k.backend.sum(e0+e1,1,keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		e = k.backend.sum(e, 3, keepdims=True)
		return e

	def call(self, x, mask=None):
		self.normalize_weights()
		xprob = x
		logker1 = -k.backend.softplus(self.kernel)
		logker0 = -k.backend.softplus(-self.kernel)
		cross = k.backend.conv2d(xprob,
		                         logker1,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		cross += k.backend.conv2d(1-xprob,
		                         logker0,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)

		ent_ker = self.ent_kernel()
		ent_x = k.backend.conv2d(xprob*k.backend.log(x),
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		ent_x += k.backend.conv2d((1-xprob) * k.backend.log(1-x),
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)

		return cross + ent_ker - ent_x

	def get_config(self):
		base_config = super(KlConv2Db, self).get_config()
		return base_config


class KlAveragePooling2D(AveragePooling2D):
	"""Average pooling operation for spatial data.

	# Arguments
		pool_size: integer or tuple of 2 integers,
			factors by which to downscale (vertical, horizontal).
			(2, 2) will halve the input in both spatial dimension.
			If only one integer is specified, the same window length
			will be used for both dimensions.
		strides: Integer, tuple of 2 integers, or None.
			Strides values.
			If None, it will default to `pool_size`.
		padding: One of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, height, width, channels)` while `channels_first`
			corresponds to inputs with shape
			`(batch, channels, height, width)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".

	# Input shape
		- If `data_format='channels_last'`:
			4D tensor with shape:
			`(batch_size, rows, cols, channels)`
		- If `data_format='channels_first'`:
			4D tensor with shape:
			`(batch_size, channels, rows, cols)`

	# Output shape
		- If `data_format='channels_last'`:
			4D tensor with shape:
			`(batch_size, pooled_rows, pooled_cols, channels)`
		- If `data_format='channels_first'`:
			4D tensor with shape:
			`(batch_size, channels, pooled_rows, pooled_cols)`
	"""

	@interfaces.legacy_pooling2d_support
	def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
	             data_format=None, **kwargs):
		super(KlAveragePooling2D, self).__init__(pool_size, strides, padding,
		                                       data_format, **kwargs)

	def _pooling_function(self, inputs, pool_size, strides,
	                      padding, data_format):
		inputs = k.backend.exp(inputs)
		output = k.backend.pool2d(inputs, pool_size, strides,
		                          padding, data_format, pool_mode='avg')
		output = k.backend.log(output)
		return output
def KlLoss(labels,preds):
	cr = -labels*preds
	ent_preds = -preds*k.backend.log(preds)
	ent_labels = 0
	cr = k.backend.tf.reduce_sum(cr,axis=[1,2,3],keep_dims=True)
	ent_preds = k.backend.tf.reduce_sum(ent_preds,axis=[1, 2, 3], keep_dims=True)
	return cr + ent_preds