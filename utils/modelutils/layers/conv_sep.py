import numpy as np
from keras.engine import InputSpec
from keras.layers import Layer, activations, initializers, constraints, regularizers,Conv2D
from keras.utils import conv_utils
from keras.legacy import interfaces
from utils.modelutils.activations.activations import *
from utils.modelutils.regularizer import regularizers as myregularizers
from utils.modelutils.regularizer.constraints import NonZero
from utils.modelutils.regularizer.initializer import VarianceScalingYingYang, compute_fans
class SplitChannelWise(Layer):
	def __init__(self,total_child,**kwargs):
		super(SplitChannelWise,self).__init__(**kwargs)
		self.total_child = total_child
	def compute_output_shape(self, input_shape):
		return self.total_child*(input_shape[0],int(input_shape[1]/self.total_child),input_shape[2],input_shape[3])
	def call(self, inputs, **kwargs):
		res = []
		filters_nb = inputs[0][1]
		for tensor in inputs:
			for i in np.arange(self.total_child):
				res+= [tensor[i*filters_nb/self.total_child:(i+1)*filters_nb/self.total_child]]
		return res
class ExpandDim(Layer):
	def __init__(self,**kwargs):
		super(ExpandDim,self).__init__(**kwargs)
	def call(self, inputs, **kwargs):
		return K.expand_dims(inputs,axis=-1)
	def compute_output_shape(self, input_shape):
		return input_shape+(1,)
class _Conv(Layer):

	def __init__(self, rank, filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
		super(_Conv, self).__init__(**kwargs)
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
		if self.rank == 1:
			outputs = K.conv1d(inputs, self.kernel, strides=self.strides[0], padding=self.padding, data_format=self.data_format,
				dilation_rate=self.dilation_rate[0])
		if self.rank == 2:
			outputs = K.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format,
				dilation_rate=self.dilation_rate)
		if self.rank == 3:
			outputs = K.conv3d(inputs, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format,
				dilation_rate=self.dilation_rate)

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
		base_config = super(_Conv, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class ConvBank(Layer):
	def __init__(self,  filters, kernel_size, strides=1, padding='valid', data_format=None, dilation_rate=1, activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None,conv_select_activation=None,select_weight_init_value=0,
	                                                                                                                                    **kwargs):
		super(ConvBank, self).__init__(**kwargs)
		rank=2
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
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.select_activation = None if conv_select_activation==None else activations.get(conv_select_activation)
		self.select_weight_init = initializers.Constant(select_weight_init_value)
		# self.input_spec = InputSpec(ndim=self.rank + 2)

	def build(self, input_shape):
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[0][channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
			                 'should be defined. Found `None`.')
		input_dim = input_shape[0][channel_axis]
		kernel_shape = self.kernel_size + (input_dim, self.filters)
		depth = int(np.log2(len(input_shape))+1)
		self.depth = depth
		population_size = len(input_shape)
		self.kernel_list = depth*[[]]
		for log2_shared_population in np.arange(depth):
			num_kernels = population_size//(2**log2_shared_population)
			for kernel_index in np.arange(num_kernels):
				self.kernel_list[log2_shared_population]= self.kernel_list[log2_shared_population]+[self.add_weight(shape=kernel_shape,
				                                                                                                    initializer=self.kernel_initializer,
				                                                                    name='kernelgroup{}_index{}'.format(log2_shared_population,kernel_index),
				                                                                                                    regularizer=self.kernel_regularizer,
				                                                                    constraint=self.kernel_constraint)]
		if not self.select_activation == None:
			self.select_weight_list = population_size*[(depth-1)*[[]]]
			weight_shape = (1,1,1,self.filters)
			for branch_index in np.arange(population_size):
				for log2_shared_population in np.arange(depth-1):

					self.select_weight_list[branch_index][log2_shared_population] = self.add_weight(shape=weight_shape,
					                                                                                initializer=self.select_weight_init,
					                                                                                name='KS_Kernel_selector_of branch{}_for '
					                                                                                     'merging Kernel{'
					                                                                                     '} to previous kernels_1 indicates '
					                                                                                     'new kernel is selected'.format(
						                                                                                branch_index, 2**log2_shared_population),
					                                                                                regularizer=None,
					                                                                                constraint=self.kernel_constraint)



		self.bias = None
		# Set input spec.
		# self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		outputs=len(inputs)*[None]
		if self.select_activation==None:
			for idx, tensor in enumerate(inputs):
				for log2_shared_population in np.arange(self.depth):
					kernel_group_index = idx//(2**log2_shared_population)
					if log2_shared_population==0:
						kernel_agg=self.kernel_list[log2_shared_population][kernel_group_index]
					else:
						kernel_agg += self.kernel_list[log2_shared_population][kernel_group_index]
				outputs[idx] = K.conv2d(tensor, kernel_agg, strides=self.strides, padding=self.padding, data_format=self.data_format,
					                   dilation_rate=self.dilation_rate)
		else:
			for idx, tensor in enumerate(inputs):
				for log2_shared_population in np.arange(self.depth-1):
					kernel_group_index = idx // (2 ** log2_shared_population)
					if log2_shared_population == 0:
						w= self.select_activation(self.select_weight_list[idx][log2_shared_population])
						kernel0 = self.kernel_list[log2_shared_population][kernel_group_index]
						kernel_append = self.kernel_list[log2_shared_population+1][kernel_group_index]
						kernel_agg = ((1-w)*kernel0)+(w*kernel_append)
					else:
						w = self.select_activation(self.select_weight_list[idx][log2_shared_population])
						kernel_append = self.kernel_list[log2_shared_population + 1][kernel_group_index]
						kernel_agg = ((1-w)*kernel_agg)+(w*kernel_append)
				outputs[idx] = K.conv2d(tensor, kernel_agg, strides=self.strides, padding=self.padding, data_format=self.data_format,
				                        dilation_rate=self.dilation_rate)
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
			space = input_shape[0][2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(space[i], self.kernel_size[i], padding=self.padding, stride=self.strides[i],
				                                        dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return len(input_shape)*[(input_shape[0][0], self.filters) + tuple(new_space)]
	def compute_mask(self, inputs, mask=None):
		return len(inputs)*[None]
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
		base_config = super(ConvBank, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
class DepthConv2D(Conv2D):
	"""Depthwise separable 2D convolution.

	Separable convolutions consist in first performing
	a depthwise spatial convolution
	(which acts on each input channel separately)
	followed by a pointwise convolution which mixes together the resulting
	output channels. The `depth_multiplier` argument controls how many
	output channels are generated per input channel in the depthwise step.

	Intuitively, separable convolutions can be understood as
	a way to factorize a convolution kernel into two smaller kernels,
	or as an extreme version of an Inception block.

	# Arguments
		filters: Integer, the dimensionality of the output space
			(i.e. the number output of filters in the convolution).
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
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
		depth_multiplier: The number of depthwise convolution output channels
			for each input channel.
			The total number of depthwise convolution output
			channels will be equal to `filterss_in * depth_multiplier`.
		activation: Activation function to use
			(see [activations](../activations.md)).
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		depthwise_initializer: Initializer for the depthwise kernel matrix
			(see [initializers](../initializers.md)).
		pointwise_initializer: Initializer for the pointwise kernel matrix
			(see [initializers](../initializers.md)).
		bias_initializer: Initializer for the bias vector
			(see [initializers](../initializers.md)).
		depthwise_regularizer: Regularizer function applied to
			the depthwise kernel matrix
			(see [regularizer](../regularizers.md)).
		pointwise_regularizer: Regularizer function applied to
			the depthwise kernel matrix
			(see [regularizer](../regularizers.md)).
		bias_regularizer: Regularizer function applied to the bias vector
			(see [regularizer](../regularizers.md)).
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation").
			(see [regularizer](../regularizers.md)).
		depthwise_constraint: Constraint function applied to
			the depthwise kernel matrix
			(see [constraints](../constraints.md)).
		pointwise_constraint: Constraint function applied to
			the pointwise kernel matrix
			(see [constraints](../constraints.md)).
		bias_constraint: Constraint function applied to the bias vector
			(see [constraints](../constraints.md)).

	# Input shape
		4D tensor with shape:
		`(batch, channels, rows, cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(batch, rows, cols, channels)` if data_format='channels_last'.

	# Output shape
		4D tensor with shape:
		`(batch, filters, new_rows, new_cols)` if data_format='channels_first'
		or 4D tensor with shape:
		`(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
		`rows` and `cols` values might have changed due to padding.
	"""

	def __init__(self, kernel_size, strides=(1, 1), padding='valid', data_format=None, depth_multiplier=1, activation=None, use_bias=True,
	             depthwise_initializer='glorot_uniform',
	             depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
	             bias_constraint=None, **kwargs):
		super(DepthConv2D, self).__init__(filters=None, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
			activation=activation, use_bias=use_bias, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
			bias_constraint=bias_constraint, **kwargs)
		self.depth_multiplier = depth_multiplier
		self.depthwise_initializer = initializers.get(depthwise_initializer)
		self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
		self.depthwise_constraint = constraints.get(depthwise_constraint)

	def build(self, input_shape):
		if len(input_shape) < 4:
			raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
			                 'Received input shape:', str(input_shape))
		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = 3
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs to '
			                 '`SeparableConv2D` '
			                 'should be defined. Found `None`.')
		input_dim = int(input_shape[channel_axis])
		depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_dim, self.depth_multiplier)

		self.depthwise_kernel = self.add_weight(shape=depthwise_kernel_shape, initializer=self.depthwise_initializer, name='depthwise_kernel',
			regularizer=self.depthwise_regularizer, constraint=self.depthwise_constraint)

		if self.use_bias:
			self.bias = self.add_weight(shape=(input_dim,), initializer=self.bias_initializer, name='bias', regularizer=self.bias_regularizer,
			                            constraint=self.bias_constraint)
		else:
			self.bias = None
		# Set input spec.
		self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
		self.built = True

	def call(self, inputs):
		outputs = K.depthwise_conv2d(inputs,self.depthwise_kernel,data_format=self.data_format,strides=self.strides,padding=self.padding)

		if self.bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)
		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			rows = input_shape[2]
			cols = input_shape[3]
		elif self.data_format == 'channels_last':
			rows = input_shape[1]
			cols = input_shape[2]

		rows = conv_utils.conv_output_length(rows, self.kernel_size[0], self.padding, self.strides[0])
		cols = conv_utils.conv_output_length(cols, self.kernel_size[1], self.padding, self.strides[1])
		if self.data_format == 'channels_first':
			return (input_shape[0], input_shape[1], rows, cols)
		elif self.data_format == 'channels_last':
			return (input_shape[0], rows, cols, input_shape[1])

	def get_config(self):
		config = super(DepthConv2D, self).get_config()
		config.pop('kernel_initializer')
		config.pop('kernel_regularizer')
		config.pop('kernel_constraint')
		config['depth_multiplier'] = self.depth_multiplier
		config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
		config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
		config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
		return config


class AmpConv2D(_Conv):

	@interfaces.legacy_conv2d_support
	def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
		super(AmpConv2D, self).__init__(rank=2, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
			dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
		self.input_spec = InputSpec(ndim=4)
	def call(self, inputs):
		input_l2 = K.sum(inputs**2,axis=[1,2,3],keepdims=True)
		outputs = K.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format,
		                   dilation_rate=self.dilation_rate)
		outputs_l2 = K.sum(outputs**2,axis=[1,2,3],keepdims=True)
		return outputs*input_l2/(outputs_l2+K.epsilon())

	def get_config(self):
		config = super(AmpConv2D, self).get_config()
		config.pop('rank')
		return config


class AmpConv2Dv1(_Conv):
	@interfaces.legacy_conv2d_support
	def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, \
	                                                                                                                               use_bias=True,
	             kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None,norm=2, **kwargs):
		super(AmpConv2Dv1, self).__init__( rank=2,filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
		                                dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
		                                bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
		                                activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
		                                bias_constraint=bias_constraint, **kwargs)
		self.input_spec = InputSpec(ndim=4)
		self.norm = norm

	def call(self, inputs):
		outputs = K.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format,
		                   dilation_rate=self.dilation_rate)
		input_l2 = (K.sum(K.abs(inputs) ** self.norm, axis=[1, 2, 3], keepdims=True))**(1/self.norm)
		outputs_l2 = (K.sum(outputs ** self.norm, axis=[1, 2, 3], keepdims=True))**(1/self.norm)
		outputs=outputs * input_l2 / (outputs_l2 + K.epsilon())
		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
		return outputs

	def get_config(self):
		config = super(AmpConv2Dv1, self).get_config()
		config.pop('rank')
		return config
class Conv3D(_Conv):
	"""3D convolution layer (e.g. spatial convolution over volumes).

	This layer creates a convolution kernel that is convolved
	with the layer input to produce a tensor of
	outputs. If `use_bias` is True,
	a bias vector is created and added to the outputs. Finally, if
	`activation` is not `None`, it is applied to the outputs as well.

	When using this layer as the first layer in a model,
	provide the keyword argument `input_shape`
	(tuple of integers, does not include the sample axis),
	e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
	with a single channel,
	in `data_format="channels_last"`.

	# Arguments
		filters: Integer, the dimensionality of the output space
			(i.e. the number output of filters in the convolution).
		kernel_size: An integer or tuple/list of 3 integers, specifying the
			depth, height and width of the 3D convolution window.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		strides: An integer or tuple/list of 3 integers,
			specifying the strides of the convolution along each spatial dimension.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Specifying any stride value != 1 is incompatible with specifying
			any `dilation_rate` value != 1.
		padding: one of `"valid"` or `"same"` (case-insensitive).
		data_format: A string,
			one of `channels_last` (default) or `channels_first`.
			The ordering of the dimensions in the inputs.
			`channels_last` corresponds to inputs with shape
			`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
			while `channels_first` corresponds to inputs with shape
			`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
			It defaults to the `image_data_format` value found in your
			Keras config file at `~/.keras/keras.json`.
			If you never set it, then it will be "channels_last".
		dilation_rate: an integer or tuple/list of 3 integers, specifying
			the dilation rate to use for dilated convolution.
			Can be a single integer to specify the same value for
			all spatial dimensions.
			Currently, specifying any `dilation_rate` value != 1 is
			incompatible with specifying any stride value != 1.
		activation: Activation function to use
			(see [activations](../activations.md)).
			If you don't specify anything, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		use_bias: Boolean, whether the layer uses a bias vector.
		kernel_initializer: Initializer for the `kernel` weights matrix
			(see [initializers](../initializers.md)).
		bias_initializer: Initializer for the bias vector
			(see [initializers](../initializers.md)).
		kernel_regularizer: Regularizer function applied to
			the `kernel` weights matrix
			(see [regularizer](../regularizers.md)).
		bias_regularizer: Regularizer function applied to the bias vector
			(see [regularizer](../regularizers.md)).
		activity_regularizer: Regularizer function applied to
			the output of the layer (its "activation").
			(see [regularizer](../regularizers.md)).
		kernel_constraint: Constraint function applied to the kernel matrix
			(see [constraints](../constraints.md)).
		bias_constraint: Constraint function applied to the bias vector
			(see [constraints](../constraints.md)).

	# Input shape
		5D tensor with shape:
		`(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if data_format='channels_first'
		or 5D tensor with shape:
		`(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if data_format='channels_last'.

	# Output shape
		5D tensor with shape:
		`(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if data_format='channels_first'
		or 5D tensor with shape:
		`(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if data_format='channels_last'.
		`new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
	"""

	@interfaces.legacy_conv3d_support
	def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None,
	             use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
	             activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
		super(Conv3D, self).__init__(rank=2, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
			dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
		self.input_spec = InputSpec(ndim=3)

	def call(self, inputs):
		if type(inputs)==list:
			tensor_list = []
			for tensor in inputs:
				tensor_list+=[K.expand_dims(tensor,axis=-1)]
			outputs = K.conv3d(tensor_list, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format,
			                   dilation_rate=self.dilation_rate)
			tensor_list=[]
			for tensor in outputs:
				tensor_list+=[K.squeeze(tensor,axis=-1)]
			outputs = tensor_list
		else:
			tensor_list = K.expand_dims(inputs,axis=-1)
			outputs = K.conv3d(tensor_list, self.kernel, strides=self.strides, padding=self.padding, data_format=self.data_format,
			                   dilation_rate=self.dilation_rate)
			outputs = K.squeeze(outputs,axis=-1)

		if self.use_bias:
			outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)

		return outputs
	def get_config(self):
		config = super(Conv3D, self).get_config()
		config.pop('rank')
		return config
