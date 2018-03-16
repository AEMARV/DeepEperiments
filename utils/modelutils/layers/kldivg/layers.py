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
from keras.layers.merge import _Merge
KER_CHAN_DIM = 2

# Merging Lobes
class AvgLobeProb(_Merge):
	"""Layer that mixes the decisions of a list of 2 inputs.

	It takes as input a list of tensors,
	all of the same shape, and returns
	a single tensor (also of the same shape).

	# Examples

	```python
		import keras

		input1 = keras.layers.Input(shape=(16,))
		x1 = keras.layers.Dense(8, activation='relu')(input1)
		input2 = keras.layers.Input(shape=(32,))
		x2 = keras.layers.Dense(8, activation='relu')(input2)
		added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

		out = keras.layers.Dense(4)(added)
		model = keras.models.Model(inputs=[input1, input2], outputs=out)
	```
	"""

	def _merge_function(self, inputs):
		if len(inputs) != 2:
			raise Exception('The input list does not have 2 tensors')
		diff = inputs[0] - inputs[1]
		output = inputs[1] + K.softplus(diff) - K.log(2.0)
		return output


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
		#self.add_loss(K.mean(-k.backend.logsumexp(x, 1, True)))
		return y
	def get_config(self):
		base_config = super(LogSoftmax, self).get_config()
		return dict(list(base_config.items()))
class NormalizeLog(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = False
		super(NormalizeLog, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		y = x/ K.sum(x, 1, True)
		y = K.clip(y,K.epsilon(),None)
		y = K.log(y)

		#self.add_loss(K.mean(-k.backend.logsumexp(x, 1, True)))
		return y
	def get_config(self):
		base_config = super(NormalizeLog, self).get_config()
		return dict(list(base_config.items()))

# Interface Base Class

# Interface Class
class KlConv2DInterface(k.layers.Conv2D):
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
		super(KlConv2DInterface, self).__init__(
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
		self.hasConcentrationPar = False
		self.concent_type = None
	def get_config(self):
		base_config = super(KlConv2DInterface, self).get_config()
		return dict(list(base_config.items()))

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
		const_shape = (kernel_shape[0],kernel_shape[1],kernel_shape[2],1)
		self.const_kernel = self.add_weight(shape=const_shape,
											initializer=k.initializers.Ones(),
											name='const_kernel',
											constraint=self.kernel_constraint,
											trainable=False,
											dtype='float32')
		if self.hasConcentrationPar:
			if self.concent_type == 'perfilter':
				concshape = (1, 1, 1, kernel_shape[3])
			elif self.concent_type=='perlayer':
				concshape = (1, 1, 1, 1)
			self.concent_par = self.add_weight(shape=concshape,
			                                    initializer=k.initializers.Constant(1),
			                                    name='concentpar',
			                                    constraint=self.kernel_constraint,
			                                    regularizer=None,
			                                    trainable=True,
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
			c = self.get_concentration()
			nkernel = self.kernel/c - k.backend.logsumexp(self.kernel/c,
														axis=KER_CHAN_DIM,
														keepdims=True)
			nkernel = K.exp(nkernel)
		else:
			nkernel = self.kernel_initializer.get_prob(self.kernel)
		return nkernel

	def get_normalizer(self):
		if not self.use_link_func:
			norm = k.backend.logsumexp(self.kernel, axis=KER_CHAN_DIM, keepdims=True)
		else:
			norm = self.kernel_initializer.get_log_normalizer(self.kernel)
		norm = K.sum(norm, axis=0,keepdims=True)
		norm = K.sum(norm, axis=1, keepdims=True)
		norm = K.sum(norm, axis=2, keepdims=True)
		norm = k.backend.permute_dimensions(norm, [0, 3, 1, 2])
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
		norm = k.backend.permute_dimensions(norm, [0, 3, 1, 2])
		return norm

	def get_bias(self):
		b = self.bias
		b = b - K.logsumexp(b, axis=0, keepdims=True)
		return b

	def get_concentration(self):
		conc = self.kernel_initializer.get_concentration(self.kernel)
		return conc

	# Scalar Graphs
	def avg_entropy(self):
		e = self.ent_per_param()
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.mean(e, 0)
		sh = K.int_shape(self.kernel)
		cat_num = sh[2]

		e = e/np.log(cat_num)
		return e

	def bias_entropy(self):
		if self.use_bias:
			b = self.get_bias()
			be = K.clip(K.exp(b), K.epsilon(), 1-K.epsilon())
			H = - b*be
			return K.sum(H)
		else:
			return -1

	def avg_concentration(self):
		conc = self.get_concentration()
		return K.max(conc)

	# Entropy
	def entropy(self):
		ent = self.ent_kernel()
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		ent = k.backend.sum(ent, 0)
		return ent

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
	# KL operations
	def kl_xl_kp(self,xl):
		pkernel = self.get_prob_kernel()
		cross_xlog_kerprob = k.backend.conv2d(xl,
											  pkernel,
											  strides=self.strides,
											  padding=self.padding,
											  data_format=self.data_format,
											  dilation_rate=self.dilation_rate)
		ent_ker = self.ent_kernel()
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
	# Log dirichlet Operations
	def get_concentrated_ent_per_component(self):
		conc_kernel = self.get_concentration()
		log_prob_kernel = self.get_log_kernel()
		return -conc_kernel*log_prob_kernel

	def get_concentrated_ent(self):
		conc_ent = self.get_concentrated_ent_per_component()
		conc_ent = K.sum(conc_ent,axis=0,keepdims=True)
		conc_ent = K.sum(conc_ent, axis=1, keepdims=True)
		conc_ent = K.sum(conc_ent, axis=2, keepdims=True)
		e = k.backend.permute_dimensions(conc_ent, [0, 3, 1, 2])
		return e

	def kl_conc_xl_kp(self,xl):
		conc_kernel = self.get_concentration()
		ent_conc_kernel = self.get_concentrated_ent()
		cross_xlog_kerprob = k.backend.conv2d(xl,
		                                      conc_kernel,
		                                      strides=self.strides,
		                                      padding=self.padding,
		                                      data_format=self.data_format,
		                                      dilation_rate=self.dilation_rate)
		return cross_xlog_kerprob+ent_conc_kernel


class KlConvBin2DInterface(k.layers.Conv2D):
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
		super(KlConvBin2DInterface, self).__init__(
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
		self.hasConcentrationPar = False
		self.concent_type = None
	def get_config(self):
		base_config = super(KlConvBin2DInterface, self).get_config()
		return dict(list(base_config.items()))

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
		const_shape = (kernel_shape[0],kernel_shape[1],kernel_shape[2],1)
		self.const_kernel = self.add_weight(shape=const_shape,
											initializer=k.initializers.Ones(),
											name='const_kernel',
											trainable=False,
											dtype='float32',
											constraint=self.kernel_constraint)
		if self.hasConcentrationPar:
			if self.concent_type == 'percomponent':
				concshape = (kernel_shape[0], kernel_shape[1], 1, kernel_shape[3])
			elif self.concent_type=='perlayer':
				concshape = (1, 1, 1, 1)
			self.concent_par = self.add_weight(shape=concshape,
			                                    initializer=k.initializers.Constant(1),
			                                    name='concentpar',
			                                    constraint=self.kernel_constraint,
			                                    regularizer=None,
			                                    trainable=True,
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

	# Weight Retrieval
	def get_log_kernel(self):
		if not self.use_link_func:
			c = self.get_concentration()
			nkernel0 = -K.softplus((self.kernel1 - self.kernel0)/c)
			nkernel1 = -K.softplus((self.kernel0 - self.kernel1)/c)
		else:
			nkernel0, nkernel1 = self.kernel_initializer.get_log_prob(self.kernel0, self.kernel1)

		return nkernel0, nkernel1

	def get_prob_kernel(self):
		if not self.use_link_func:
			c= self.get_concentration()
			nkernel0 = K.sigmoid((self.kernel0 - self.kernel1)/c)
			nkernel1 = K.sigmoid((self.kernel1 - self.kernel0)/c)
		else:
			nkernel0, nkernel1 = self.kernel_initializer.get_prob(self.kernel0, self.kernel1)

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
		z = k.backend.permute_dimensions(z, [0, 3, 1, 2])
		return K.exp(z)

	def get_log_normalizer(self):
		if not self.use_link_func:
			z = self.kernel0 + K.softplus(self.kernel1 - self.kernel0)

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel0,self.kernel1)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		z = k.backend.permute_dimensions(z, [0, 3, 1, 2])
		return z

	def get_concentration(self):
		conc0,conc1 = self.kernel_initializer.get_concentration(self.kernel0, self.kernel1)

		return conc0,conc1

	# Entropy
	def entropy(self):
		e = self.ent_kernel()
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)
		e = k.backend.sum(e, 0)

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
	# Scalar Graphs
	def bias_entropy(self):
		if self.use_bias:
			b = self.get_bias()
			be = K.clip(K.exp(b), K.epsilon(), 1-K.epsilon())
			H = - b*be
			return K.sum(H)
		else:
			return -1

	def avg_entropy(self):
		e = self.ent_per_param()
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = k.backend.mean(e, 0)
		e = e/np.log(2)
		return e

	def avg_concentration(self):
		conc0, conc1 = self.get_concentration()
		return K.max(conc0/2+conc1/2)

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

	# Log dirichlet Operations
	def get_concentrated_ent_per_component(self):
		conc_kernel0, conc_kernel1 = self.get_concentration()
		log_kernel0, log_kernel1 = self.get_log_kernel()
		return -(conc_kernel0*log_kernel0 + conc_kernel1*log_kernel1)

	def get_concentrated_ent(self):
		conc_ent = self.get_concentrated_ent_per_component()
		conc_ent = K.sum(conc_ent, axis=0, keepdims=True)
		conc_ent = K.sum(conc_ent, axis=1, keepdims=True)
		conc_ent = K.sum(conc_ent, axis=2, keepdims=True)
		e = k.backend.permute_dimensions(conc_ent, [0, 3, 1, 2])
		return  e

	def kl_conc_xl_kp(self,x):
		xprob = x
		xprob = k.backend.clip(xprob, k.backend.epsilon(), 1 - k.backend.epsilon())
		logx1 = k.backend.log(xprob)
		logx0 = k.backend.log(1 - xprob)
		conc_kernel0,conc_kernel1 = self.get_concentration()
		ent_conc_kernel = self.get_concentrated_ent()
		cross_xlog_kerp = k.backend.conv2d(logx1,
		                                   conc_kernel1,
		                                   strides=self.strides,
		                                   padding=self.padding,
		                                   data_format=self.data_format,
		                                   dilation_rate=self.dilation_rate)
		cross_xlog_kerp += k.backend.conv2d(logx0,
		                                    conc_kernel0,
		                                    strides=self.strides,
		                                    padding=self.padding,
		                                    data_format=self.data_format,
		                                    dilation_rate=self.dilation_rate)
		return cross_xlog_kerp + ent_conc_kernel


#Legacy
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
		return dict(list(base_config.items()))

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
		if not self.use_link_func:
			kernel0 = -K.softplus(self.kernel)
			kernel1 = -K.softplus(-self.kernel)
		else:
			kernel0,kernel1 = self.kernel_initializer.get_log_prob(self.kernel)
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
		z = k.backend.permute_dimensions(z, [0, 3, 1, 2])
		return K.exp(z)

	def get_log_normalizer(self):
		if not self.use_link_func:
			z = self.kernel*0

		else:
			z = self.kernel_initializer.get_log_normalizer(self.kernel)
		z = K.sum(z, axis=0, keepdims=True)
		z = K.sum(z, axis=1, keepdims=True)
		z = K.sum(z, axis=2, keepdims=True)
		z = k.backend.permute_dimensions(z, [0, 3, 1, 2])
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

# Normalized KLs
class KlConv2D(KlConv2DInterface):

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


class KlConvLogit2D(_KlConvLogit2D):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConvLogit2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def get_config(self):
		base_config = super(KlConvLogit2D, self).get_config()
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


class KlConvBin2D(KlConvBin2DInterface):
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
			out = K.bias_add(out, self.bias, data_format=self.data_format)
		return out

	def get_config(self):
		base_config = super(KlConvBin2D, self).get_config()
		return base_config


# KL Biased and Concentrated
class KlConv2D_Concentrated(KlConv2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Concentrated, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.hasConcentrationPar = True


	def get_config(self):
		base_config = super(KlConv2D_Concentrated, self).get_config()
		return base_config

	def call(self, x, mask=None):
		out = self.kl_xl_kp(x)
		out = out * (K.exp(self.concent_par))
		#outI= self.kl_xp_kl(x)*(K.softplus(self.concent_par_input))
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out


class KlConv2Db_Concentrated(KlConvBin2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2Db_Concentrated, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.hasConcentrationPar = True


	def call(self, x, mask=None):

		out = self.kl_xl_kp(x)
		out = out * (K.exp(self.concent_par))
		#outI = self.kl_xp_kl(x) * (K.softplus(self.concent_par_input))
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out

	def get_config(self):
		base_config = super(KlConv2Db_Concentrated, self).get_config()
		return base_config
# Naturally Concentrated
class KlConv2D_NConcentrated(KlConv2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_NConcentrated, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.hasConcentrationPar = False
		self.use_link_func = True
	def kl_conc_xp_kl(self,xp):
		lkernel = self.get_log_kernel()
		xl = k.backend.log(xp)
		cross_xprob_kerlog = k.backend.conv2d(xp,
											  lkernel,
											  strides=self.strides,
											  padding=self.padding,
											  data_format=self.data_format,
											  dilation_rate=self.dilation_rate)
		ent_x = k.backend.conv2d(-xp * xl,
								 self.const_kernel,
								 strides=self.strides,
								 padding=self.padding,
								 data_format=self.data_format,
								 dilation_rate=self.dilation_rate)
		return cross_xprob_kerlog + ent_x
	def get_prob_kernel(self):
		conc = self.get_concentration()
		kernelsz = K.shape(self.kernel)
		kernelsz = float(kernelsz[0]*kernelsz[1])
		normal_conc_par = (self.get_concentration())**(1/kernelsz)
		conc = conc/normal_conc_par
		conc = conc/K.sum(conc, axis=2, keepdims=True)
		return conc
	def get_log_kernel(self):
		return K.log(self.get_prob_kernel())
	def call(self, x, mask=None):
		out = self.kl_conc_xp_kl(x)
		if self.use_bias:
			out = K.bias_add(out, self.bias, data_format=self.data_format)
		return out


class KlConv2Db_NConcentrated(KlConvBin2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2Db_NConcentrated, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.hasConcentrationPar = False
		self.use_link_func = True


	def call(self, x, mask=None):

		out = self.kl_conc_xl_kp(x)
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out

# Concentrated, and Single Component Filters
class KlConv2DSC(KlConv2DInterface):
	''' Single Component Convolution:
	Kl convolution with natural parameters
	The input patch is treated as a distribution on the field where the spatial and channel locations are different
	 states
	'''
	def __init__(self,
				 filters,
				 kernel_size,
	             isinput=False,
				 **kwargs):
		super(KlConv2DSC, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.hasConcentrationPar = False
		self.use_link_func = True
		self.is_input= isinput
	def get_concentrated_ent(self):
		kerlog = self.get_log_kernel()
		conc = self.get_concentration()
		conc_ent = conc*kerlog
		conc_ent = K.sum(conc_ent, axis=0, keepdims=True)
		conc_ent = K.sum(conc_ent, axis=1, keepdims=True)
		conc_ent = K.sum(conc_ent, axis=2, keepdims=True)
		e = k.backend.permute_dimensions(conc_ent, [0, 3, 1, 2])
		return e
	def get_normalizer(self):
		norm = self.kernel_initializer.get_normalizer(self.kernel)
		norm = k.backend.permute_dimensions(norm, [0, 3, 1, 2])
		return norm
	# Weight Retrieval
	def kl_xl_kp_sc(self,x):
		# Concentrated single component KL divergence
		if self.is_input:
			x= K.clip(x, K.epsilon(), None)
			xl = K.log(x)
		else:
			xl = x
			x = K.exp(x)
			x = K.clip(x, K.epsilon(), None)
		conc_kernel = self.get_concentration()
		cross_xlog_kerconc = k.backend.conv2d(xl,
											  conc_kernel,
											  strides=self.strides,
											  padding=self.padding,
											  data_format=self.data_format,
											  dilation_rate=self.dilation_rate)
		norm_x = k.backend.conv2d(x,
		                          self.const_kernel,
		                          strides=self.strides,
		                          padding=self.padding,
		                          data_format=self.data_format,
		                          dilation_rate=self.dilation_rate)

		ent_conc_ker = self.get_concentrated_ent()
		KLD = cross_xlog_kerconc + ent_conc_ker - (K.log(norm_x)*self.get_normalizer())
		return KLD
	def kl_xp_kl_sc(self,x,xl):
		# Concentrated single component KL divergence
		log_kernel = self.get_log_kernel()
		cross_xlog_kerconc = k.backend.conv2d(x,
											  log_kernel,
											  strides=self.strides,
											  padding=self.padding,
											  data_format=self.data_format,
											  dilation_rate=self.dilation_rate)
		norm_x = k.backend.conv2d(x,
		                          self.const_kernel,
		                          strides=self.strides,
		                          padding=self.padding,
		                          data_format=self.data_format,
		                          dilation_rate=self.dilation_rate)
		ent_x = k.backend.conv2d(-x * xl,
		                         self.const_kernel,
		                         strides=self.strides,
		                         padding=self.padding,
		                         data_format=self.data_format,
		                         dilation_rate=self.dilation_rate)
		KLD = cross_xlog_kerconc + ent_x + K.log(norm_x)*norm_x
		return KLD
	def call(self, x, mask=None):
		if self.is_input:
			x= K.clip(x,K.epsilon(),None)
			xp = x
			xl = K.log(x)
		else:
			xp = K.exp(x)
			xl = x
		out = self.kl_xp_kl_sc(xp,xl)
		if self.use_bias:
			out = K.bias_add(out, self.bias, data_format=self.data_format)
		#out = out - K.logsumexp(out,axis=1,keepdims=True)
		return out
# Double Lobed Convs
class KlConvLobed2D(KlConv2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConvLobed2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.empathic = True
		self.igo_centric = True
	def get_empathic(self):
		self.empathic = True
		self.igo_centric = False
	def get_igo_centric(self):
		self.empathic = False
		self.igo_centric = True
	def get_igo_empathic(self):
		self.empathic = True
		self.igo_centric=True
	def call(self, x, mask=None):
		out = 0
		KLD = self.kl_xl_kp(x)
		KLDI = self.kl_xp_kl(x)
		if self.empathic:
			out += KLD
		if self.igo_centric:
			out += KLDI
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out
class KlConvLobedBin2D(KlConvBin2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConvLobedBin2D, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)
		self.empathic = True
		self.igo_centric = True
	def get_empathic(self):
		self.empathic = True
		self.igo_centric = False
	def get_igo_centric(self):
		self.empathic = False
		self.igo_centric = True
	def get_igo_empathic(self):
		self.empathic = True
		self.igo_centric=True
	def call(self, x, mask=None):
		out = 0
		KLD = self.kl_xl_kp(x)
		KLDI = self.kl_xp_kl(x)
		if self.empathic:
			out += KLD
		if self.igo_centric:
			out += KLDI
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out

# Unnormalized
class KlConv2D_Unnorm(KlConv2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Unnorm, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def call(self, x, mask=None):
		KLD = self.kl_xl_kp(x)
		KLDI = self.kl_xp_kl(x)
		kernel_norm = self.get_normalizer()
		out = KLD*kernel_norm
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out
class KlConv2D_Unnorm_Bin(KlConvBin2DInterface):

	def __init__(self,
				 filters,
				 kernel_size,
				 **kwargs):
		super(KlConv2D_Unnorm_Bin, self).__init__(
			filters=filters,
			kernel_size=kernel_size,
			**kwargs)

	def call(self, x, mask=None):
		KLD = self.kl_xl_kp(x)
		KLDI = self.kl_xp_kl(x)
		kernel_norm = self.get_normalizer()
		out = KLD*kernel_norm
		if self.use_bias:
			out = K.bias_add(out, self.get_bias(), data_format=self.data_format)
		return out

# Bregman
class KlConv2D_Breg(KlConv2DInterface):

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
class KlConv2D_Breg_Un_Norm(KlConv2DInterface):

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




