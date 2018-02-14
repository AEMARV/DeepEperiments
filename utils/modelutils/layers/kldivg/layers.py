from keras.layers import Layer
import keras as k
import tensorflow as tf
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.engine import InputSpec
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
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
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
		e = self.kernel * k.backend.exp(self.kernel)
		e = k.backend.sum(e,4,keepdims=True)
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
class KlConv2Db()