from keras.layers import Layer
import keras as k
import numpy as np
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

KER_CHAN_DIM = 2


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
#TODO in normalizers please fix the permute thingy

# Interface Class
class _KlConv2D(k.layers.Conv2D):
	def __init__(self, filters,
	             kernel_size,
	             strides=(1, 1),
	             padding='valid',
	             data_format=None,
	             dilation_rate=(1, 1),
	             activation=None,
	             use_bias=None,
	             kernel_initializer=None,
	             bias_initializer='zeros',
	             kernel_regularizer=None,
	             bias_regularizer=None,
	             activity_regularizer=None,
	             kernel_constraint=None,
	             bias_constraint=None,
	             dist_measure=None,
	             use_link_func=None,
	             **kwargs):
		super(_KlConv2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			data_format='channels_first',
			dilation_rate=dilation_rate,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			**kwargs)
		self.use_link_func = use_link_func
		self.dist_measure = dist_measure

	def get_config(self):
		base_config = super(_KlConv2D, self).get_config()
		return base_config

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

	# Weight Retrieval
	def get_log_kernel(self):

		if not self.use_link_func:
			nkernel = self.kernel - k.backend.logsumexp(self.kernel,
															axis=KER_CHAN_DIM,
															keepdims=True)
		else:
			nkernel = self.kernel_initializer.get_log_prob(self.kernel)
		return nkernel

	def get_prob_kernel(self):

		if not self.use_link_func:
			nkernel = self.kernel - k.backend.logsumexp(self.kernel,
			                                            axis=KER_CHAN_DIM,
			                                            keepdims=True)
			nkernel = K.exp(nkernel)
		else:
			nkernel = self.kernel_initializer.get_prob(self.kernel)
		return nkernel

	def get_normalizer(self):
		if not self.use_link_func:
			norm = K.exp(k.backend.logsumexp(self.kernel, axis=KER_CHAN_DIM, keepdims=True))
		else:
			norm = self.kernel_initializer.get_log_normalizer(self.kernel)
		norm = K.sum(norm,axis=0,keepdims=True)
		norm = K.sum(norm, axis=1, keepdims=True)
		norm = K.sum(norm, axis=2, keepdims=True)
		return K.exp(norm)

	def get_log_normalizer(self):
		if not self.use_link_func:
			norm = k.backend.logsumexp(self.kernel,
															axis=KER_CHAN_DIM,
															keepdims=True)
		else:
			norm = self.kernel_initializer.get_log_normalizer(self.kernel)
		norm = K.sum(norm, axis=0, keepdims=True)
		norm = K.sum(norm, axis=1, keepdims=True)
		norm = K.sum(norm, axis=2, keepdims=True)
		return norm

	def get_bias(self):
		b = self.bias
		b = b - K.logsumexp(b, axis=0, keepdims=True)
		return b

	# Entropy
	def entropy(self):
		ent = self.ent_kernel()
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		return ent

	def avg_entropy(self):
		e = self.ent_per_param()
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		sh = K.int_shape(self.kernel)
		cat_num = sh[2]

		e = e/np.log(cat_num)
		return e

	def ent_per_param(self):
		lkernel = self.get_log_kernel()
		pkernel = self.get_prob_kernel()
		e = -lkernel * pkernel
		return e

	def ent_kernel(self):
		e = self.ent_per_param()
		e = k.backend.sum(e, 0, keepdims=True)
		e = k.backend.sum(e, 1, keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		e = k.backend.permute_dimensions(e, [0, 3, 1, 2])
		return e

	#OPS
	def kl_xl_kp(self,xl):
		pkernel = self.get_prob_kernel()
		cross_xlog_kerprob = k.backend.conv2d(xl,
		                                      pkernel,
		                                      strides=self.strides,
		                                      padding=self.padding,
		                                      data_format=self.data_format,
		                                      dilation_rate=self.dilation_rate)
		ent_ker = self.ent_kernel()
		ent_ker = k.backend.permute_dimensions(ent_ker, [0, 3, 1, 2])
		return cross_xlog_kerprob + ent_ker
	def kl_xp_kl(self,xl):
		lkernel = self.get_log_kernel()
		xprob = k.backend.exp(xl)
		cross_xprob_kerlog = k.backend.conv2d(xprob,
		                                      lkernel,
		                                      strides=self.strides,
		                                      padding=self.padding,
		                                      data_format=self.data_format,
		                                      dilation_rate=self.dilation_rate)
		ent_x = k.backend.conv2d(-xprob * xl,
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		return cross_xprob_kerlog + ent_x


class _KlConvLogit2D(k.layers.Conv2D):
	def __init__(self, filters,
	             kernel_size,
	             strides=(1, 1),
	             padding='valid',
	             data_format=None,
	             dilation_rate=(1, 1),
	             activation=None,
	             use_bias=None,
	             kernel_initializer=None,
	             bias_initializer='zeros',
	             kernel_regularizer=None,
	             bias_regularizer=None,
	             activity_regularizer=None,
	             kernel_constraint=None,
	             bias_constraint=None,
	             dist_measure=None,
	             use_link_func=None,
	             **kwargs):
		super(_KlConvLogit2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			data_format='channels_first',
			dilation_rate=dilation_rate,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			**kwargs)
		self.use_link_func = use_link_func
		self.dist_measure = dist_measure

	def get_config(self):
		base_config = super(_KlConvLogit2D, self).get_config()
		return base_config

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

	# Weight Retrieval
	def get_log_kernel(self):
		kernel0 = -K.softplus(self.kernel)
		kernel1 = -K.softplus(-self.kernel)
		return kernel0, kernel1

	def get_prob_kernel(self):
		kernel0 = K.sigmoid(-self.kernel)
		kernel1 = K.sigmoid(self.kernel)
		return kernel0, kernel1

	def get_bias(self):
		b = self.bias
		b = b - K.logsumexp(b,axis=0,keepdims=True)
		return b

	def get_normalizer(self):
		if not self.use_link_func:
			z = self.kernel*0

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		return K.exp(z)

	def get_log_normalizer(self):
		if not self.use_link_func:
			z = self.kernel*0

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		return z

	# Entropy Retrieval
	def entropy(self):
		e = self.ent_kernel()
		e = k.backend.sum(e,0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		return e

	def avg_entropy(self):
		e = self.ent_per_param()
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.sum(e, 0)
		sh = K.int_shape(self.kernel)
		e = e/np.log(2)
		return e

	def ent_per_param(self):
		lkernel0, lkernel1 = self.get_log_kernel()
		pkernel0, pkernel1 = self.get_prob_kernel()
		e1 = -pkernel1 * lkernel1
		e1 += -pkernel0 * lkernel0
		return e1

	def ent_kernel(self):
		'''Ent Kernel calculates the entropy of each kernel -PlogP
		note that the - sign is implicit in softplus'''
		e = self.ent_per_param()
		e = k.backend.sum(e, 0, keepdims=True)
		e = k.backend.sum(e, 1, keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		e = k.backend.permute_dimensions(e, [0, 3, 1, 2])
		return e

	#OPS
	def kl_xl_kp(self,x):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1-k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1-xprob)
		pker0, pker1 = self.get_prob_kernel()
		ent_ker = self.ent_kernel()
		cross_xlog_kerp = k.backend.conv2d(logx1,
		                                   pker1,
		                                   strides=self.strides,
		                                   padding=self.padding,
		                                   data_format=self.data_format,
		                                   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
		                                    pker0,
		                                    strides=self.strides,
		                                    padding=self.padding,
		                                    data_format=self.data_format,
		                                    dilation_rate=self.dilation_rate)
		return cross_xlog_kerp + ent_ker

	def kl_xp_kl(self,x):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1 - k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1 - xprob)
		logker0, logker1 = self.get_log_kernel()
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
		code_length_x = xprob * logx1
		code_lenght_nx = (1 - xprob) * logx0
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
		return cross_xp_kerlog + ent_x


class _KlConvBin2D(k.layers.Conv2D):
	def __init__(self, filters,
	             kernel_size,
	             strides=(1, 1),
	             padding='valid',
	             data_format=None,
	             dilation_rate=(1, 1),
	             activation=None,
	             use_bias=None,
	             kernel_initializer=None,
	             bias_initializer='zeros',
	             kernel_regularizer=None,
	             bias_regularizer=None,
	             activity_regularizer=None,
	             kernel_constraint=None,
	             bias_constraint=None,
	             dist_measure=None,
	             use_link_func=None,
	             **kwargs):
		super(_KlConvBin2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			data_format='channels_first',
			dilation_rate=dilation_rate,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer,
			kernel_constraint=kernel_constraint,
			bias_constraint=bias_constraint,
			**kwargs)
		self.use_link_func = use_link_func
		self.dist_measure = dist_measure

	def get_config(self):
		base_config = super(_KlConvBin2D, self).get_config()
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
		self.kernel1 = self.add_weight(shape=kernel_shape,
									  initializer=self.kernel_initializer,
									  name='kernel1',
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)
		self.kernel0 = self.add_weight(shape=kernel_shape,
									   initializer=self.kernel_initializer,
									   name='kernel0',
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

	# Weight Retrieval
	def get_log_kernel(self):
		if not self.use_link_func:
			nkernel0 = -K.softplus(self.kernel1 - self.kernel0)
			nkernel1 = -K.softplus(self.kernel0 - self.kernel1)
		else:
			nkernel0,nkernel1 = self.kernel_initializer.get_log_prob(self.kernel0,self.kernel1)

		return nkernel0, nkernel1

	def get_prob_kernel(self):
		if not self.use_link_func:
			nkernel0 = K.sigmoid(self.kernel0 - self.kernel1)
			nkernel1 = K.sigmoid(self.kernel1 - self.kernel0)
		else:
			nkernel0,nkernel1 = self.kernel_initializer.get_prob(self.kernel0,self.kernel1)

		return nkernel0, nkernel1

	def get_bias(self):
		b = self.bias
		b = b - K.logsumexp(b, axis=0, keepdims=True)
		return b

	def get_normalizer(self):
		if not self.use_link_func:
			z = self.kernel0 + K.softplus(self.kernel1 - self.kernel0)

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel0,self.kernel1)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		return K.exp(z)

	def get_log_normalizer(self):
		if not self.use_link_func:
			z = self.kernel0 + K.softplus(self.kernel1 - self.kernel0)

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel0,self.kernel1)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		return z

	# Entropy
	def entropy(self):
		e = self.ent_kernel()
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)

		return e

	def avg_entropy(self):
		e = self.ent_per_param()
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.sum(e, 0)
		sh = K.int_shape(self.kernel)
		e = e/np.log(2)
		return e

	def ent_per_param(self):
		lker0, lker1 = self.get_log_kernel()
		pker0, pker1 = self.get_prob_kernel()
		e = -(lker0 * pker0) - (lker1 * pker1)
		return e

	def ent_kernel(self):
		'''Ent Kernel calculates the entropy of each kernel -PlogP
		note that the - sign is implicit in softplus'''
		e = self.ent_per_param()
		e = k.backend.sum(e, 0, keepdims=True)
		e = k.backend.sum(e, 1, keepdims=True)
		e = k.backend.sum(e, 2, keepdims=True)
		e = k.backend.permute_dimensions(e, [0, 3, 1, 2])
		return e

	# OPS
	def kl_xl_kp(self, x):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1 - k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1 - xprob)
		pker0, pker1 = self.get_prob_kernel()
		cross_xlog_kerp = k.backend.conv2d(logx1,
		                                   pker1,
		                                   strides=self.strides,
		                                   padding=self.padding,
		                                   data_format=self.data_format,
		                                   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
		                                    pker0,
		                                    strides=self.strides,
		                                    padding=self.padding,
		                                    data_format=self.data_format,
		                                    dilation_rate=self.dilation_rate)

		ent_ker = self.ent_kernel()

		return cross_xlog_kerp + ent_ker

	def kl_xp_kl(self, x):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1 - k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1 - xprob)
		logker0, logker1 = self.get_log_kernel()
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

		code_length_x = xprob * logx1
		code_lenght_nx = (1 - xprob) * logx0
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
		return cross_xp_kerlog + ent_x


# Normalized KLs
class KlConv2D(_KlConv2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConv2D, self).get_config()
		return base_config

	def call(self, xl, mask=None):
		lkernel = self.get_log_kernel()
		pkernel = self.get_prob_kernel()
		xprob = k.backend.exp(xl)
		cross_xprob_kerlog = k.backend.conv2d(xprob,
								 lkernel,
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		cross_xlog_kerprob = k.backend.conv2d(xl,
											  pkernel,
											  strides=self.strides,
											  padding=self.padding,
											  data_format=self.data_format,
											  dilation_rate=self.dilation_rate)

		ent_ker = self.ent_kernel()
		ent_x = k.backend.conv2d(-xprob*xl,
								 self.const_kernel,
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		out = self.dist_measure(cross_xprob_kerlog, cross_xlog_kerprob, ent_x, ent_ker)
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out


class KlConv2Db(_KlConvLogit2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2Db, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConv2Db, self).get_config()
		return base_config

	def call(self, x, mask=None):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1-k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1-xprob)
		logker0,logker1 = self.get_log_kernel()
		pker0 , pker1 = self.get_prob_kernel()
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
										   pker1,
										   strides=self.strides,
										   padding=self.padding,
										   data_format=self.data_format,
										   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
										   pker0,
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


		out = self.dist_measure(cross_xp_kerlog, cross_xlog_kerp, ent_x, ent_ker)
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out


class KlConvBin2D(_KlConvBin2D):
	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConvBin2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def call(self, x, mask=None):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1-k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1-xprob)
		logker0,logker1 = self.get_log_kernel()
		pker0, pker1 = self.get_prob_kernel()
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
										   pker1,
										   strides=self.strides,
										   padding=self.padding,
										   data_format=self.data_format,
										   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
										   pker0,
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

		out = self.dist_measure(cross_xp_kerlog, cross_xlog_kerp, ent_x, ent_ker)
		if self.use_bias:
			out = K.bias_add(out,self.bias,data_format=self.data_format)
		return out

	def get_config(self):
		base_config = super(KlConvBin2D, self).get_config()
		return base_config


# KL Biased and Concentrated
class KlConv2D_Concentrated(_KlConv2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Concentrated, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)


	def get_config(self):
		base_config = super(KlConv2D_Concentrated, self).get_config()
		return base_config

	def call(self, x, mask=None):
		nkernel = self.get_log_kernel()
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

		out = cross_xlog_kerprob + ent_ker
		out = K.abs(self.concentration) * out
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out


class KlConv2Db_Concentrated(_KlConvLogit2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2Db_Concentrated, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)


	def call(self, x, mask=None):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1-k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1-xprob)
		logker0,logker1 = self.get_log_kernel()
		pker0, pker1= self.get_prob_kernel()
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
										   pker1,
										   strides=self.strides,
										   padding=self.padding,
										   data_format=self.data_format,
										   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
										   pker0,
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

		out = cross_xlog_kerp + ent_ker
		out = out * K.abs(self.concentration)
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out

	def get_config(self):
		base_config = super(KlConv2Db_Concentrated, self).get_config()
		return base_config


# Unnormalized
class KlConv2D_Un_Norm(_KlConv2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Un_Norm, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConv2D_Breg_Un_Norm, self).get_config()
		return base_config

	def call(self, x, mask=None):
		KLD =self.kl_xl_kp(x)
		normalizer = self.get_normalizer()*KLD

		return out
class KlConv2D_Un_Norm(_KlConv2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Un_Norm, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConv2D_Breg_Un_Norm, self).get_config()
		return base_config


	def get_normalizer(self):

		if not self.use_link_func:
			z = K.logsumexp(self.kernel, axis=2, keepdims=True)
		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		return K.exp(z)
	def get_log_normalizer(self):

		if not self.use_link_func:
			z = K.logsumexp(self.kernel, axis=2, keepdims=True)
		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		return z

	def call(self, x, mask=None):
		nkernel = self.get_log_kernel()
		normalizer = self.get_normalizer()
		expnkernel = K.exp(nkernel)
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
		ker_sum = K.sum(expnkernel, axis=2, keepdims=True)
		ker_sum = K.log(ker_sum)
		ker_sum = K.sum(ker_sum, axis=0, keepdims=True)
		ker_sum = K.sum(ker_sum, axis=1, keepdims=True)
		ker_sum = K.exp(ker_sum)
		ker_sum = k.backend.permute_dimensions(ker_sum, [0, 3, 1, 2])
		ent_x = k.backend.conv2d(-xprob * x,
								 self.const_kernel,
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		data_sum = k.backend.sum(xprob, axis=1, keepdims=True)
		data_sum = k.backend.log(data_sum)
		a = self.const_kernel[:, :, 0:1, :]
		data_sum = k.backend.conv2d(data_sum,
									self.const_kernel[:, :, 0:1, 0:1],
									strides=self.strides,
									padding=self.padding,
									data_format=self.data_format,
									dilation_rate=self.dilation_rate)
		data_sum = K.exp(data_sum)


		normalizer = k.backend.permute_dimensions(normalizer, [0, 3, 1, 2])
		out = (cross_xlog_kerprob + ent_ker)*normalizer# +ker_sum - data_sum


		return out

# Bregman
class KlConv2D_Breg(_KlConv2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Breg, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConv2D_Breg, self).get_config()
		return base_config

	def call(self, x, mask=None):
		nkernel = self.get_log_kernel()
		expnkernel = K.exp(nkernel)
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
		ker_sum = K.sum(expnkernel,axis=2,keepdims=True)
		ker_sum = K.log(ker_sum)
		ker_sum = K.sum(ker_sum,axis=0,keepdims=True)
		ker_sum = K.sum(ker_sum,axis =1,keepdims=True)
		ker_sum = K.exp(ker_sum)
		ker_sum = k.backend.permute_dimensions(ker_sum, [0, 3, 1, 2])
		ent_x = k.backend.conv2d(-xprob*x,
								 self.const_kernel,
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		data_sum = k.backend.sum(xprob,axis=1,keepdims=True)
		data_sum = k.backend.log(data_sum)
		a = self.const_kernel[:,:,0:1,:]
		data_sum = k.backend.conv2d(data_sum,
								 self.const_kernel[:,:,0:1,0:1],
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		data_sum = K.exp(data_sum)

		out = (cross_xlog_kerprob + ent_ker) + ker_sum - data_sum
		return out


class KlConv2D_Logit_Breg(_KlConvLogit2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Logit_Breg, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_normalizer(self):
		if not self.use_link_func:
			z = self.kernel0 + K.softplus(self.kernel1 - self.kernel0)

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		return K.exp(z)

	def get_log_normalizer(self):
		if not self.use_link_func:
			z = self.kernel0 + K.softplus(self.kernel1 - self.kernel0)

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		return z

	def call(self, x, mask=None):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1-k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1-xprob)
		logker0,logker1 = self.get_log_kernel()
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
										   k.backend.exp(logker1),
										   strides=self.strides,
										   padding=self.padding,
										   data_format=self.data_format,
										   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
										   k.backend.exp(logker0),
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

		sumker = K.exp(logker0) + K.exp(logker1)
		sumker = K.prod(sumker, axis=0,keepdims=True)
		sumker = K.prod(sumker, axis=1, keepdims=True)
		sumker = K.prod(sumker, axis=2, keepdims=True)
		sumker = k.backend.permute_dimensions(sumker, [0, 3, 1, 2])
		normalizer = k.backend.permute_dimensions(normalizer, [0, 3, 1, 2])
		out = (cross_xlog_kerp + ent_ker)*normalizer#+sumker -1

		return out

	def get_config(self):
		base_config = super(KlConv2D_Logit_Breg, self).get_config()
		return base_config


# LOGITS
class KlConv2D_Breg_Un_Norm(_KlConv2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Breg_Un_Norm, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConv2D_Breg_Un_Norm, self).get_config()
		return base_config

	def get_normalizer(self):
		if not self.use_link_func:

			z = K.logsumexp(self.kernel, axis=2,keepdims=True)
		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.exp(z)
		return z

	def get_log_normalizer(self):
		if not self.use_link_func:

			z = K.logsumexp(self.kernel, axis=2,keepdims=True)
		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		return z

	def call(self, x, mask=None):
		nkernel = self.get_log_kernel()
		normalizer = self.get_normalizer()
		expnkernel = K.exp(nkernel)
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
		ker_sum = K.sum(expnkernel, axis=2, keepdims=True)
		ker_sum = K.log(ker_sum)
		ker_sum = K.sum(ker_sum, axis=0, keepdims=True)
		ker_sum = K.sum(ker_sum, axis=1, keepdims=True)
		ker_sum = K.exp(ker_sum)
		ker_sum = k.backend.permute_dimensions(ker_sum, [0, 3, 1, 2])
		ent_x = k.backend.conv2d(-xprob * x,
								 self.const_kernel,
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		data_sum = k.backend.sum(xprob, axis=1, keepdims=True)
		data_sum = k.backend.log(data_sum)
		a = self.const_kernel[:, :, 0:1, :]
		data_sum = k.backend.conv2d(data_sum,
									self.const_kernel[:, :, 0:1, 0:1],
									strides=self.strides,
									padding=self.padding,
									data_format=self.data_format,
									dilation_rate=self.dilation_rate)
		data_sum = K.exp(data_sum)

		normalizer = k.backend.permute_dimensions(normalizer, [0, 3, 1, 2])
		out = (cross_xlog_kerprob + ent_ker)*normalizer# +ker_sum - data_sum


		return out


## POOLING
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




