from keras.layers import Layer
import keras as k
import tensorflow as tf
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from utils.modelutils.layers.kldivg.initializers import *
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.layers.pooling import AveragePooling2D
from keras.backend import epsilon
from keras.legacy import interfaces

KER_CHAN_DIM =2
class LogSoftmax(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = False
		super(LogSoftmax, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		y = x - k.backend.logsumexp(x, 1, True)
		return y

	def get_config(self):
		base_config = super(LogSoftmax, self).get_config()
		return dict(list(base_config.items()))


class KlConv2D(k.layers.Conv2D):

	def __init__(self,
	             filters,
	             kernel_size,
	             rank=2,
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
		super(KlConv2D, self).__init__(filters,kernel_size,**kwargs)
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
		                              initializer=Softmax_Init(),
		                              name='kernel',
		                              regularizer=self.kernel_regularizer,
		                              constraint=self.kernel_constraint)
		self.const_kernel = self.add_weight(shape=kernel_shape,
		                                    initializer=k.initializers.Ones(),
		                                    name='const_kernel',
		                                    constraint=self.kernel_constraint,
		                                    trainable=False,
		                                    dtype='float32')
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


		self.built = True

	def get_config(self):
		base_config = super(KlConv2D, self).get_config()
		return base_config

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

	def compute_mask(self, input, input_mask=None):
		return None

	def normalize_weights(self):
		nkernel = self.kernel - k.backend.logsumexp(self.kernel,
		                                                axis=KER_CHAN_DIM,
		                                                keepdims=True)
		return nkernel

	def entropy(self):
		ent = self.ent_kernel()
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		return ent

	def ent_kernel(self):
		nkernel = self.normalize_weights()
		e = -nkernel * k.backend.exp(nkernel)
		e = k.backend.sum(e, 0, keepdims=True)
		e = k.backend.sum(e, 1, keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		return e

	def call(self, x, mask=None):
		nkernel = self.normalize_weights()
		xprob = k.backend.exp(x)
		cross_xprob_kerlog = k.backend.conv2d(xprob,
		                         nkernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		cross_xlog_kerprob = k.backend.conv2d(x,
		                                      k.backend.exp(nkernel),
		                                      strides=self.strides,
		                                      padding=self.padding,
		                                      data_format=self.data_format,
		                                      dilation_rate=self.dilation_rate)

		ent_ker = self.ent_kernel()
		ent_x = k.backend.conv2d(-xprob*x,
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		ent_ker = k.backend.permute_dimensions(ent_ker, [0, 3, 1, 2])
		y = calc_dist(cross_xprob_kerlog, cross_xlog_kerprob, ent_x, ent_ker)
		return y


class KlConv2Db(k.layers.Conv2D):
	def __init__(self,
	             filters,
	             kernel_size,
	             rank=2,
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
		super(KlConv2Db, self).__init__(filters, kernel_size,**kwargs)
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
		                              initializer=Sigmoid_Init(),
		                              name='kernel',
		                              regularizer=self.kernel_regularizer,
		                              constraint=self.kernel_constraint)
		self.const_kernel = self.add_weight(shape=kernel_shape,
		                                    initializer=k.initializers.Ones(),
		                                    name='const_kernel',
		                                    trainable=False,
		                                    dtype='float32',
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

		self.built = True

	def compute_mask(self, input, input_mask=None):
		return None

	def entropy(self):
		e = self.ent_kernel()
		e = k.backend.sum(e,0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)

		return e

	def ent_kernel(self):
		e1 = k.backend.sigmoid(self.kernel)*k.backend.softplus(-self.kernel)
		e0 = k.backend.sigmoid(-self.kernel)*k.backend.softplus(self.kernel)
		e = k.backend.sum(e0 + e1, 0, keepdims=True)
		e = k.backend.sum(e, 1, keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		return e

	def call(self, x, mask=None):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1-k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1-xprob)
		logker1 = -k.backend.softplus(-self.kernel)
		logker0 = -k.backend.softplus(self.kernel)
		cross_xp_kerlog = k.backend.conv2d(xprob,
		                                   logker1,
		                                   strides=self.strides,
		                                   padding=self.padding,
		                                   data_format=self.data_format,
		                                   dilation_rate=self.dilation_rate)
		cross_xp_kerlog += k.backend.conv2d(1 - xprob,
				                            logker0,
				                            strides=self.strides,
				                            padding=self.padding,
				                            data_format=self.data_format,
				                            dilation_rate=self.dilation_rate)
		cross_xlog_kerp = k.backend.conv2d(logx1,
				                           k.backend.sigmoid(self.kernel),
				                           strides=self.strides,
				                           padding=self.padding,
				                           data_format=self.data_format,
				                           dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
		                                   k.backend.sigmoid(-self.kernel),
		                                   strides=self.strides,
		                                   padding=self.padding,
		                                   data_format=self.data_format,
		                                   dilation_rate=self.dilation_rate)

		ent_ker = self.ent_kernel()
		code_length_x = xprob*logx1
		code_lenght_nx = (1-xprob)*logx0
		ent_x = k.backend.conv2d(code_length_x,
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		ent_x += k.backend.conv2d(code_lenght_nx,
		                          self.const_kernel,
		                          strides=self.strides,
		                          padding=self.padding,
		                          data_format=self.data_format,
		                          dilation_rate=self.dilation_rate)
		ent_ker = k.backend.permute_dimensions(ent_ker, [0, 3, 1, 2])

		return calc_dist(cross_xp_kerlog, cross_xlog_kerp, ent_x, ent_ker)

	def get_config(self):
		base_config = super(KlConv2Db, self).get_config()
		return base_config


class KlAveragePooling2D(AveragePooling2D):

	def _pooling_function(self, inputs, pool_size, strides,
	                      padding, data_format):
		inputs = k.backend.exp(inputs)
		output = k.backend.pool2d(inputs, pool_size, strides,
		                          padding, data_format, pool_mode='avg')
		output = k.backend.clip(output, k.backend.epsilon(), 1-k.backend.epsilon())
		#output = output/k.backend.sum(output,axis=KER_CHAN_DIM,keepdims=True)
		output = k.backend.log(output)
		return output


def calc_dist(cross_xprob_kerlog, cross_xlog_kerprob, ent_x, ent_ker):
	distance = 0
	distance += cross_xlog_kerprob
	distance += cross_xprob_kerlog
	distance += ent_x
	distance += ent_ker
	return distance


def kl_loss(y_true,y_pred):
	cr = y_true*y_pred
	ent_preds = -k.backend.exp(y_pred)*y_pred
	ent_labels = 0
	cr = k.backend.sum(cr, axis=[-1])
	ent_preds = k.backend.sum(ent_preds, axis=[-1])
	calc_dist(0, cr, ent_preds, 0)
	return -cr
