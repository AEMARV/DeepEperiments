import numpy as np
from keras import initializers
from keras.layers import Dropout
from keras.layers import Layer,activations,initializers,constraints,regularizers
from numpy.testing.tests.test_utils import my_cacw

from utils.modelutils.regularizer import regularizers as myregularizers
from utils.modelutils.regularizer.constraints import NonZero
from utils.modelutils.regularizer.initializer import VarianceScalingYingYang,compute_fans
from keras.engine import InputSpec
from keras.utils import conv_utils
import keras.backend as K


from utils.modelutils.activations.activations import *
from utils.modelutils.regularizer.entropy_activity_reg import SoftmaxEntropyRegularizer
from keras.layers.convolutional import _Conv


class Conv2DRandom(Layer):
	def __init__(self, filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, ying_yang=True, **kwargs):
		super(Conv2DYingYang, self).__init__(**kwargs)
		rank = 2
		self.rank = 2
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
		# self.kernel_regularizer = myregularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(ndim=self.rank + 2)
		self.kernel_regularizer = kernel_regularizer
		self.ying_yang = ying_yang

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
			                 'should be defined. Found `None`.')
		input_dim = input_shape[channel_axis]
		kernel_shape = self.kernel_size + (input_dim, 2 * self.filters)
		kernel_ying_shape = self.kernel_size + (input_dim, self.filters)
		kernel_yang_shape = self.kernel_size + (input_dim, self.filters)
		self.kernel_ying_shape = kernel_ying_shape
		self.kernel_yang_shape = kernel_yang_shape

		self.kernel_shape = kernel_shape
		self.kernel_ying = self.add_weight(shape=kernel_ying_shape, initializer=self.kernel_initializer, name='kernel',
		                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, trainable=self.ying_yang)
		self.kernel_yang = self.add_weight(shape=kernel_yang_shape, initializer=self.kernel_initializer, name='kernel',
		                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, trainable=not self.ying_yang)
		if self.use_bias:
			self.bias_ying = self.add_weight(shape=(self.filters,), initializer=self.bias_initializer, name='bias',
			                                 regularizer=self.bias_regularizer,
			                                 constraint=self.bias_constraint, trainable=self.ying_yang)
			self.bias_yang = self.add_weight(shape=(self.filters,), initializer=self.bias_initializer, name='bias',
			                                 regularizer=self.bias_regularizer,
			                                 constraint=self.bias_constraint, trainable=not self.ying_yang)
		else:
			self.bias = None

		# Set input spec.
		self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		fan_in, fan_out = _compute_fans(shape)
		scale = self.scale
		if self.mode == 'fan_in':
			scale /= max(1., fan_in)
		elif self.mode == 'fan_out':
			scale /= max(1., fan_out)
		else:
			scale /= max(1., float(fan_in + fan_out) / 2)
		if self.distribution == 'normal':
			stddev = np.sqrt(scale)
			return K.truncated_normal(shape, 0., stddev, dtype=dtype, seed=self.seed)
		# kernel_size_mul = np.muself.kernel_shape
		K.random_uniform()
		outputs = K.conv2d(inputs, K.concatenate([self.kernel_ying, self.kernel_yang], axis=-1), strides=self.strides, padding=self.padding,
		                   data_format=self.data_format, dilation_rate=self.dilation_rate)

		if self.use_bias:
			outputs = K.bias_add(outputs, K.concatenate([self.bias_ying, self.bias_yang], axis=-1), data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (2 * self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], 2 * self.filters) + tuple(new_space)

	def get_config(self):
		config = {
			'rank'                : self.rank,
			'filters'             : self.filters,
			'kernel_size'         : self.kernel_size,
			'strides'             : self.strides,
			'padding'             : self.padding,
			'data_format'         : self.data_format,
			'dilation_rate'       : self.dilation_rate,
			'activation'          : activations.serialize(self.activation),
			'use_bias'            : self.use_bias,
			'kernel_initializer'  : initializers.serialize(self.kernel_initializer),
			'bias_initializer'    : initializers.serialize(self.bias_initializer),
			'kernel_regularizer'  : regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer'    : regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint'   : constraints.serialize(self.kernel_constraint),
			'bias_constraint'     : constraints.serialize(self.bias_constraint)
		}
		base_config = super(Conv2DYingYang, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class Conv2DYingYang(Layer):
	def __init__(self, filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, ying_yang=True, **kwargs):
		super(Conv2DYingYang, self).__init__(**kwargs)
		rank = 2
		self.rank = 2
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
		# self.kernel_regularizer = myregularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(ndim=self.rank + 2)
		self.kernel_regularizer = kernel_regularizer
		self.ying_yang = ying_yang

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
		kernel_ying_shape = self.kernel_size+(input_dim,self.filters)
		kernel_yang_shape = self.kernel_size+(input_dim,self.filters)
		self.kernel_ying_shape = kernel_ying_shape
		self.kernel_yang_shape = kernel_yang_shape

		self.kernel_shape = kernel_shape
		self.kernel_ying = self.add_weight(shape=kernel_ying_shape, initializer=VarianceScalingYingYang(scale=2.0), name='kernel_ying',
		                                   regularizer=self.kernel_regularizer,
		                              constraint=self.kernel_constraint,trainable=self.ying_yang)
		self.kernel_yang = self.add_weight(shape=kernel_yang_shape, initializer=VarianceScalingYingYang(scale=2.0), name='kernel_yang',
		                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,trainable=self.ying_yang)
		if self.use_bias:
			self.bias = super(Conv2DYingYang,self).add_weight(shape=(self.filters,), initializer=self.bias_initializer, name='bias',
			                                  regularizer=self.bias_regularizer,
			                            constraint=self.bias_constraint,trainable=True)
		else:
			self.bias = None

		# Set input spec.
		self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		# kernel_size_mul = np.muself.kernel_shape
		kernel = K.in_train_phase(K.concatenate([self.kernel_ying, self.kernel_yang],axis=2),self.kernel_ying)
		input = K.in_train_phase(K.concatenate([inputs, inputs], axis=1),
		                         inputs)
		outputs = K.conv2d(input,kernel, strides=self.strides, padding=self.padding,
		                   data_format=self.data_format, dilation_rate=self.dilation_rate)

		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (2*self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], self.filters) + tuple(new_space)

	def get_config(self):
		config = {
			'rank'                : self.rank,
			'filters'             : self.filters,
			'kernel_size'         : self.kernel_size,
			'strides'             : self.strides,
			'padding'             : self.padding,
			'data_format'         : self.data_format,
			'dilation_rate'       : self.dilation_rate,
			'activation'          : activations.serialize(self.activation),
			'use_bias'            : self.use_bias,
			'kernel_initializer'  : initializers.serialize(self.kernel_initializer),
			'bias_initializer'    : initializers.serialize(self.bias_initializer),
			'kernel_regularizer'  : regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer'    : regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint'   : constraints.serialize(self.kernel_constraint),
			'bias_constraint'     : constraints.serialize(self.bias_constraint)
		}
		base_config = super(Conv2DYingYang, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Conv2DRandomYang(Layer):
	def __init__(self, filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, ying_yang=True, **kwargs):
		super(Conv2DRandomYang, self).__init__(**kwargs)
		rank = 2
		self.rank = 2
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
		# self.kernel_regularizer = myregularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(ndim=self.rank + 2)
		self.kernel_regularizer = kernel_regularizer
		self.ying_yang = ying_yang

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
		kernel_ying_shape = self.kernel_size + (input_dim, self.filters)
		kernel_yang_shape = self.kernel_size + (input_dim, self.filters)
		self.kernel_ying_shape = kernel_ying_shape
		self.kernel_yang_shape = kernel_yang_shape

		self.kernel_shape = kernel_shape
		self.kernel_ying = self.add_weight(shape=kernel_ying_shape, initializer=VarianceScalingYingYang(scale=2.0), name='kernel_ying',
		                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, trainable=self.ying_yang)
		self.kernel_yang = self.add_weight(shape=kernel_yang_shape, initializer=VarianceScalingYingYang(scale=2.0), name='kernel_yang',
		                                   regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, trainable=not self.ying_yang)
		if self.use_bias:
			self.bias = super(Conv2DRandomYang, self).add_weight(shape=(self.filters,), initializer=self.bias_initializer, name='bias',
			                                                   regularizer=self.bias_regularizer, constraint=self.bias_constraint, trainable=True)
		else:
			self.bias = None
		scale = 2.0
		fan_in,fan_out= compute_fans(kernel_yang_shape)
		scale /= max(1., 2*fan_in)
		self.stddev = np.sqrt(scale)
		# Set input spec.
		self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		# kernel_size_mul = np.muself.kernel_shape

		noise_yang = K.truncated_normal(K.shape(self.kernel_ying), 0., self.stddev)
		kernel = K.in_train_phase(K.concatenate([self.kernel_ying, noise_yang], axis=2), K.concatenate([self.kernel_ying, self.kernel_yang], axis=2))
		input = K.concatenate([inputs, inputs], axis=1)
		outputs = K.conv2d(input, kernel, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)

		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (2 * self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], self.filters) + tuple(new_space)

	def get_config(self):
		config = {
			'rank'                : self.rank,
			'filters'             : self.filters,
			'kernel_size'         : self.kernel_size,
			'strides'             : self.strides,
			'padding'             : self.padding,
			'data_format'         : self.data_format,
			'dilation_rate'       : self.dilation_rate,
			'activation'          : activations.serialize(self.activation),
			'use_bias'            : self.use_bias,
			'kernel_initializer'  : initializers.serialize(self.kernel_initializer),
			'bias_initializer'    : initializers.serialize(self.bias_initializer),
			'kernel_regularizer'  : regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer'    : regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint'   : constraints.serialize(self.kernel_constraint),
			'bias_constraint'     : constraints.serialize(self.bias_constraint)
		}
		base_config = super(Conv2DRandomYang, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class Conv2DTanh(Layer):
	def __init__(self,  filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None,kernel_max_init = 1,bias_max_init=1, **kwargs):
		super(Conv2DTanh, self).__init__(**kwargs)
		rank =2
		self.rank = 2
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
		# self.kernel_regularizer = myregularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.input_spec = InputSpec(ndim=self.rank + 2)
		self.kernel_max_init = kernel_max_init
		self.bias_max_init = bias_max_init
		self.kernel_regularizervals =kernel_regularizer


	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
			                 'should be defined. Found `None`.')
		input_dim = input_shape[channel_axis]
		self.max_weight = self.add_weight(shape=(self.filters,), initializer=initializers.Constant(self.kernel_max_init),constraint=NonZero(),
		name='max_weight')
		self.bias_max = self.add_weight(shape=(self.filters,), initializer=initializers.Constant(self.bias_max_init), name='bias_max')
		self.bias_slop = self.add_weight(shape=(self.filters,), initializer=initializers.Ones(), name='bias_slope')
		self.slope = self.add_weight(shape=(self.filters,), initializer=initializers.Constant(1/self.kernel_max_init), name='slope')
		self.kernel_regularizer = myregularizers.l1_l2_tanh(l1=self.kernel_regularizervals.l1, l2=self.kernel_regularizervals.l2, layer=self)
		kernel_shape = self.kernel_size + (input_dim, self.filters)
		self.kernel_shape = kernel_shape
		self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer, name='kernel', regularizer=self.kernel_regularizer,
		                              constraint=self.kernel_constraint)
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,), initializer=self.bias_initializer, name='bias', regularizer=self.bias_regularizer,
			                            constraint=self.bias_constraint)
		else:
			self.bias = None

		# Set input spec.
		self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self.built = True


	def call(self, inputs):
		# kernel_size_mul = np.muself.kernel_shape
		outputs = K.conv2d(inputs,self.max_weight*tanh(self.kernel*self.slope), strides=self.strides, padding=self.padding,
		                   data_format=self.data_format,
			dilation_rate=self.dilation_rate)

		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias_max*tanh(self.bias*self.bias_slop), data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], self.filters) + tuple(new_space)


	def get_config(self):
		config = {
			'rank'                : self.rank,
			'filters'             : self.filters,
			'kernel_size'         : self.kernel_size,
			'strides'             : self.strides,
			'padding'             : self.padding,
			'data_format'         : self.data_format,
			'dilation_rate'       : self.dilation_rate,
			'activation'          : activations.serialize(self.activation),
			'use_bias'            : self.use_bias,
			'kernel_initializer'  : initializers.serialize(self.kernel_initializer),
			'bias_initializer'    : initializers.serialize(self.bias_initializer),
			'kernel_regularizer'  : regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer'    : regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint'   : constraints.serialize(self.kernel_constraint),
			'bias_constraint'     : constraints.serialize(self.bias_constraint)
		}
		base_config = super(Conv2DTanh,self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

