import warnings

# from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Conv2D, Activation, AveragePooling2D, Lambda, MaxPooling2D, BatchNormalization, Merge, \
	SeparableConv2D, MaxoutDense
from keras.layers.merge import add, average, concatenate, multiply
from keras.regularizers import l1_l2

from utils.modelutils.layers.binary_layers.birelu import *
from utils.modelutils.layers.conv import *
from utils.modelutils.layers.conv_sep import *
from utils.modelutils.regularizer.regularizers import l1_l2_tanh
from keras.models import Model
from utils.modelutils.layers.activations import *
from utils.modelutils.layers.kldivg.layers import *
from utils.modelutils.activations import activations as cact

layer_index = 0
# use of | will make the parser for load weight not considering string past '|'
CONVSH_NAME = 'BLOCK{}_CONV'
KL_CONV_NAME = 'KLCONV{}'
KL_CONVB_NAME = 'KLCONVB{}'
CONV_NAME = 'BLOCK{}_CONV-{}'
ACT_NAME_RULE = 'BLOCK{}_ACT_{}{}'
POOL_NAME_RULE = 'BLOCK{}_POOL_{}'
BATCH_NORMSH_RULE = 'BLOCK{}_BATCHNORM'
BATCH_NORM_RULE = 'BLOCK{}_BATCHNORM_{}'


# COMPONENT_NAMES
def Layer_on_list(layer, tensor_list):
	res = []
	tensor_list = node_list_to_list(tensor_list)
	for x in tensor_list:
		res += [layer(x)]
	return res


def get_layer_index():
	global layer_index
	layer_index = layer_index + 1
	return layer_index


def model_constructor(opts, model_dict=None):
	'nb_filter_list is total filters used in each layer.filter size is for convolution'

	img_input = Input(shape=opts['training_opts']['dataset']['input_shape'], name='image_batch')
	x = [[img_input]]
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	block_index = 0
	queue_dict = {}
	yang_aux_phase = StateLayer()
	for layer in model_dict:
		component = layer['type']

		param = layer['param']
		with tf.name_scope(component):
			if component == 'push':
				name = param['name']
				queue_dict[name] = x
			#------------------------------------------------------Kl Layers

			elif component == 'lsoft':
				x = Layer_on_list(LogSoftmax(), x)
			# Normalized KL CONVS
			elif component == 'klconv':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['kl_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D(filters=int(nb_filter * expand_rate),
				                           kernel_size=kernel_size,
				                           padding=padding,
										   kernel_initializer=init,
										   kernel_regularizer=reg,
										   activation=None,
				                           use_link_func=use_link_func,
				                           dist_measure=dist_measure,
										   name=KL_CONV_NAME.format(block_index),
										   use_bias=True), x)
			elif component == 'klconvl':

				"""Generic Setting"""
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['klb_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']

				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convbreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConvLogit2D(filters=int(nb_filter * expand_rate),
				                            kernel_size=kernel_size,
				                            padding=padding,
										    kernel_initializer=init,
										    kernel_regularizer=reg,
										    activation=None,
										    name=KL_CONVB_NAME.format(block_index),
				                            use_link_func=use_link_func,
				                            dist_measure=dist_measure,
										    use_bias=True), x)
			elif component == 'klconvb':

				"""Generic Setting"""
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['klb_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']

				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convbreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConvBin2D(filters=int(nb_filter * expand_rate),
				                            kernel_size=kernel_size,
				                            padding=padding,
										    kernel_initializer=init,
										    kernel_regularizer=reg,
										    activation=None,
										    name=KL_CONVB_NAME.format(block_index),
				                            use_link_func=use_link_func,
				                            dist_measure=dist_measure,
										    use_bias=True), x)
			# Unnormalized KL Conv
			elif component == 'klconvu':
				# TODO fix this block, nothing changed here
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['kl_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D_Unnorm(filters=int(nb_filter * expand_rate),
				                           kernel_size=kernel_size,
				                           padding=padding,
										   kernel_initializer=init,
										   kernel_regularizer=reg,
										   activation=None,
				                           use_link_func=use_link_func,
				                           dist_measure=dist_measure,
										   name=KL_CONV_NAME.format(block_index),
										   use_bias=True), x)
			elif component == 'klconvbu':
				# TODO fix this block, nothing changed here
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['klb_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convbreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D_Unnorm_Bin(filters=int(nb_filter * expand_rate),
				                           kernel_size=kernel_size,
				                           padding=padding,
										   kernel_initializer=init,
										   kernel_regularizer=reg,
										   activation=None,
				                           use_link_func=use_link_func,
				                           dist_measure=dist_measure,
										   name=KL_CONV_NAME.format(block_index),
										   use_bias=True), x)
			# Kl Conv with Concentration Parametrization
			elif component == 'klconvconc':

				# KL conv with Concentration
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['kl_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D_Concentrated(filters=int(nb_filter * expand_rate),
				                           kernel_size=kernel_size,
				                           padding=padding,
										   kernel_initializer=init,
										   kernel_regularizer=reg,
										   activation=None,
				                           use_link_func=use_link_func,
				                           dist_measure=dist_measure,
										   name=KL_CONV_NAME.format(block_index),
										   use_bias=True), x)
			elif component == 'klconvbreg':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 0)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['kl_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D_Breg(filters=int(nb_filter * expand_rate),
				                           kernel_size=kernel_size,
				                           padding=padding,
										   kernel_initializer=init,
										   kernel_regularizer=reg,
										   activation=None,
				                           use_link_func=use_link_func,
				                           dist_measure=dist_measure,
										   name=KL_CONV_NAME.format(block_index),
										   use_bias=False), x)
			elif component == 'klconvbregunnorm':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['kl_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D_Breg_Un_Norm(filters=int(nb_filter * expand_rate),
				                           kernel_size=kernel_size,
				                           padding=padding,
										   kernel_initializer=init,
										   kernel_regularizer=reg,
										   activation=None,
				                           use_link_func=use_link_func,
				                           dist_measure=dist_measure,
										   name=KL_CONV_NAME.format(block_index),
										   use_bias=False), x)

			elif component == 'klconvbconc':

				# KL conv with Concentration
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['kl_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']
				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2Db_Concentrated(filters=int(nb_filter * expand_rate),
				                                        kernel_size=kernel_size,
				                                        padding=padding,
				                                        kernel_initializer=init,
				                                        kernel_regularizer=reg,
				                                        activation=None,
				                                        use_link_func=use_link_func,
				                                        dist_measure=dist_measure,
				                                        name=KL_CONV_NAME.format(block_index),
				                                        use_bias=True), x)
			elif component == 'klconvbSP':

				"""Generic Setting"""
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['klb_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']

				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convbreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2Db_Sep_Filt(filters=int(nb_filter * expand_rate),
				                            kernel_size=kernel_size,
				                            padding=padding,
										    kernel_initializer=init,
										    kernel_regularizer=reg,
										    activation=None,
										    name=KL_CONVB_NAME.format(block_index),
				                            use_link_func=use_link_func,
				                            dist_measure=dist_measure,
										    use_bias=True), x)
			elif component == 'klconvbSPbreg':

				"""Generic Setting"""
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				padding = param['padding'] if 'padding' in param else 'same'

				"""INIT"""
				init = opts['model_opts']['kl_opts']['klb_initial']

				"""Distance Measure"""
				dist_measure = opts['model_opts']['kl_opts']['dist_measure']

				"""Weight Encoding"""
				use_link_func = opts['model_opts']['kl_opts']['use_link_func']

				"""Regularization"""
				reg = opts['model_opts']['kl_opts']['convbreg']
				if reg is not None:
					reg_coef = param['coef'] if 'coef' in param else 1
					reg = reg(coef=reg_coef)
					reg.use_link_func = use_link_func
					reg.link_func = init.linkfunc
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(KlConv2D_Logit_Breg(filters=int(nb_filter * expand_rate),
				                            kernel_size=kernel_size,
				                            padding=padding,
										    kernel_initializer=init,
										    kernel_regularizer=reg,
										    activation=None,
										    name=KL_CONVB_NAME.format(block_index),
				                            use_link_func=use_link_func,
				                            dist_measure=dist_measure,
										    use_bias=True), x)
			elif component == 'klavgpool':
				pool_size = int(param['r'])
				strides = int(param['s'])
				pool_size = (pool_size, pool_size)
				strides = (strides, strides)
				padding = param['pad'] if 'pad' in param.keys() else 'valid'
				x = node_list_to_list(x)
				x = Layer_on_list(
					KlAveragePooling2D(pool_size=pool_size,
					                   strides=strides,
					                   padding=padding,
					                   name=POOL_NAME_RULE.format(block_index, 'AVERAGE')), x)
			## End KL Layers
			elif component == 'convsh':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(Conv2D(filters=int(nb_filter * expand_rate),
				                         kernel_size=kernel_size,
				                         padding=padding,
										 kernel_initializer=initializion,
										 kernel_regularizer=w_reg,
										 activation=activation,
										 name=CONVSH_NAME.format(block_index),
										 use_bias=use_bias),
				                  x)
			elif component == 'convshfixedfilter':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Conv2D(filters=int(nb_filter), kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
						   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias), x)
			elif component == 'dconv':
				block_index += 1
				mult = int(param['mult']) if 'mult' in param else 1
				nb_filter = int(param['f'])
				iterations = int(param['iter'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				conv_layer = DepthConv2D(kernel_size=kernel_size, padding=padding, depthwise_regularizer=w_reg, depthwise_initializer=initializion,
										 name='DCONV{}'.format(block_index), depth_multiplier=mult)
				x = node_list_to_list(x)
				y = x
				res = []
				for i in np.arange(iterations):
					y = Layer_on_list(conv_layer, y)
					res += y
				x = res
				x = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=(1, 1), padding=padding, kernel_initializer=initializion,
										 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index),
										 use_bias=use_bias), x)
			elif component == 'dconvconcat':
				block_index += 1
				mult = int(param['mult']) if 'mult' in param else 1
				nb_filter = int(param['f'])
				iterations = int(param['iter'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				conv_layer = DepthConv2D(kernel_size=kernel_size, padding=padding, depthwise_regularizer=w_reg, depthwise_initializer=initializion,
										 name='DCONV{}'.format(block_index), depth_multiplier=mult)
				x = node_list_to_list(x)
				y = x
				res = x
				for i in np.arange(iterations):
					y = Layer_on_list(conv_layer, y)
					res2 = []
					for idx, tensor in enumerate(y):
						res2 += [
							BatchNormalization(axis=1, epsilon=1e-5, name=(BATCH_NORM_RULE + 'DCONV_ITER{}').format(block_index, idx, i))(tensor)]
					y = res2
					res += y
				x = res
				x = [concatenate(x, axis=1)]
				x = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=(1, 1), padding=padding, kernel_initializer=initializion,
										 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index),
										 use_bias=use_bias), x)
			elif component == 'dconvberconcat':
				block_index += 1
				mult = int(param['mult']) if 'mult' in param else 1
				nb_filter = int(param['f'])
				iterations = int(param['iter'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				conv_layer = DepthConv2D(kernel_size=kernel_size, padding=padding, depthwise_regularizer=w_reg, depthwise_initializer=initializion,
										 name='DCONV{}'.format(block_index), depth_multiplier=mult)
				x = node_list_to_list(x)
				y = x
				res = x
				for i in np.arange(iterations):
					y = Layer_on_list(conv_layer, y)
					res2 = []
					for idx, tensor in enumerate(y):
						bn_tensor = BatchNormalization(axis=1, epsilon=1e-5, name=(BATCH_NORM_RULE + 'DCONV_ITER{}').format(block_index, idx, i))(
							tensor)
						bn_tensor_ber = Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'BER_INNER{}'.format(i), idx))(bn_tensor)
						res2 += bn_tensor_ber
					y = res2
					res += y
				x = res
				x = [concatenate(x, axis=1)]
				x = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=(1, 1), padding=padding, kernel_initializer=initializion,
										 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index),
										 use_bias=use_bias), x)
			elif component == 'bnsh':
				x = node_list_to_list(x)
				x = Layer_on_list(BatchNormalization(axis=1, epsilon=1e-5, name=BATCH_NORMSH_RULE.format(block_index)), x)
			elif component == 'bn':
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					res += [BatchNormalization(axis=1, epsilon=1e-5, name=BATCH_NORM_RULE.format(block_index, idx))(tensor)]
				x = res
			elif component == 'conv':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for idx, tensor in enumerate(x):
						res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									   kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(
							tensor)]
				x = res
			elif component == 'convsns':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				shratio = param['shratio']
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				convshared = Conv2D(filters=int(nb_filter * expand_rate * shratio), kernel_size=kernel_size, padding=padding,
									kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
									name=CONVSH_NAME.format(block_index), use_bias=use_bias)
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for idx, tensor in enumerate(x):
						ns_tensor = Conv2D(filters=int(nb_filter * expand_rate * (1 - shratio)), kernel_size=kernel_size, padding=padding,
										   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
										   name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(tensor)
						s_tensor = convshared(tensor)
						s_ns = concatenate([s_tensor, ns_tensor], axis=1)
						res += [s_ns]

				x = res
			elif component == 'convsnsrec':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				con_type_total = int(np.log2(len(x)) + 1)
				population_size = len(x)
				conv_bank = con_type_total * [[]]
				if len(x) == 1:
					res += [Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					f_size = 2 ** (-con_type_total + 1)
					for log2_shared_population in np.arange(con_type_total):
						num_kernels = population_size // (2 ** log2_shared_population)
						set_size = (population_size // num_kernels)
						for kernel_index in np.arange(num_kernels):
							instance_begin = set_size * kernel_index
							conv_bank[log2_shared_population] = conv_bank[log2_shared_population] + [
								Conv2D(filters=int(nb_filter * expand_rate * (f_size)), kernel_size=kernel_size, padding=padding,
									   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
									   name=CONVSH_NAME.format(block_index) + 'CONV_REC_MEBMBERSIZE_{}_INSTANCE_{}_TO_{}'.format(
										   2 ** log2_shared_population, instance_begin, instance_begin + set_size), use_bias=use_bias)]
						if not log2_shared_population == 0:
							f_size = f_size * 2
						res = []
					for idx, tensor in enumerate(x):
						tensor_to_merge = []
						for log2_shared_population in np.arange(con_type_total):
							kernel_group_index = idx // (2 ** log2_shared_population)
							tensor_to_merge += [conv_bank[log2_shared_population][kernel_group_index](tensor)]
						res += [concatenate(tensor_to_merge, axis=1)]

				x = res
			elif component == 'convsnsrecshratio':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				shratio = param['shratio']
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				con_type_total = int(np.log2(len(x)) + 1)
				population_size = len(x)
				conv_bank = con_type_total * [[]]
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for log2_shared_population in np.arange(con_type_total):
						num_kernels = population_size // (2 ** log2_shared_population)
						if log2_shared_population == 0:
							f_size = ((1 - shratio) ** (con_type_total - log2_shared_population - 1)) / num_kernels
						else:
							f_size = ((1 - shratio) ** (con_type_total - log2_shared_population - 1)) * shratio / num_kernels
						set_size = (population_size // num_kernels)
						for kernel_index in np.arange(num_kernels):
							instance_begin = set_size * kernel_index
							conv_bank[log2_shared_population] = conv_bank[log2_shared_population] + [
								Conv2D(filters=int((nb_filter * expand_rate * (f_size)) + 1), kernel_size=kernel_size, padding=padding,
									   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
									   name=CONVSH_NAME.format(block_index) + 'CONV_REC_MEBMBERSIZE_{}_INSTANCE_{}_TO_{}'.format(
										   2 ** log2_shared_population, instance_begin, instance_begin + set_size), use_bias=use_bias)]
						if not log2_shared_population == 0:
							f_size = f_size * 2
						res = []
					for idx, tensor in enumerate(x):
						tensor_to_merge = []
						for log2_shared_population in np.arange(con_type_total):
							kernel_group_index = idx // (2 ** log2_shared_population)
							tensor_to_merge += [conv_bank[log2_shared_population][kernel_group_index](tensor)]
						res += [concatenate(tensor_to_merge, axis=1)]

				x = res
			elif component == 'convsnsrecshratiorev':
				# allocation ratio is reversed. meaning 1-p category gets the first portion and will distrubute the filters to number of kernels
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				shratio = param['shratio']
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				con_type_total = int(np.log2(len(x)) + 1)
				population_size = len(x)
				conv_bank = con_type_total * [[]]
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for log2_shared_population in np.arange(con_type_total):
						num_kernels = population_size // (2 ** log2_shared_population)
						if log2_shared_population == con_type_total - 1:
							f_size = ((1 - shratio) ** (log2_shared_population)) / num_kernels
						else:
							f_size = ((1 - shratio) ** (log2_shared_population)) * shratio / num_kernels
						set_size = (population_size // num_kernels)
						for kernel_index in np.arange(num_kernels):
							instance_begin = set_size * kernel_index
							conv_bank[log2_shared_population] = conv_bank[log2_shared_population] + [
								Conv2D(filters=int(nb_filter * expand_rate * (f_size)), kernel_size=kernel_size, padding=padding,
									   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
									   name=CONVSH_NAME.format(block_index) + 'CONV_REC_MEBMBERSIZE_{}_INSTANCE_{}_TO_{}'.format(
										   2 ** log2_shared_population, instance_begin, instance_begin + set_size), use_bias=use_bias)]
						if not log2_shared_population == 0:
							f_size = f_size * 2
						res = []
					for idx, tensor in enumerate(x):
						tensor_to_merge = []
						for log2_shared_population in np.arange(con_type_total):
							kernel_group_index = idx // (2 ** log2_shared_population)
							tensor_to_merge += [conv_bank[log2_shared_population][kernel_group_index](tensor)]
						res += [concatenate(tensor_to_merge, axis=1)]

				x = res
			elif component == 'convsnsrecratiofixed':
				# if we have 8 branches each category of kernels(in this case 4) reserve 1/(total categories) so each kernel is 1/4*number of
				# kernels in the category
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				con_type_total = int(np.log2(len(x)) + 1)
				population_size = len(x)
				conv_bank = con_type_total * [[]]
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for log2_shared_population in np.arange(con_type_total):
						num_kernels = population_size // (2 ** log2_shared_population)
						f_size = 1 / (num_kernels * con_type_total)
						set_size = (population_size // num_kernels)
						for kernel_index in np.arange(num_kernels):
							instance_begin = set_size * kernel_index
							conv_bank[log2_shared_population] = conv_bank[log2_shared_population] + [
								Conv2D(filters=int(nb_filter * expand_rate * (f_size)), kernel_size=kernel_size, padding=padding,
									   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
									   name=CONVSH_NAME.format(block_index) + 'CONV_REC_MEBMBERSIZE_{}_INSTANCE_{}_TO_{}'.format(
										   2 ** log2_shared_population, instance_begin, instance_begin + set_size), use_bias=use_bias)]
						if not log2_shared_population == 0:
							f_size = f_size * 2
						res = []
					for idx, tensor in enumerate(x):
						tensor_to_merge = []
						for log2_shared_population in np.arange(con_type_total):
							kernel_group_index = idx // (2 ** log2_shared_population)
							tensor_to_merge += [conv_bank[log2_shared_population][kernel_group_index](tensor)]
						res += [concatenate(tensor_to_merge, axis=1)]

				x = res
			elif component == 'convsnsdy':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				shratio = param['shratio']
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				convshared = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for idx, tensor in enumerate(x):
						# res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
						#                kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx),
						# use_bias=use_bias)(
						# 	tensor)]
						ns_tensor = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding,
										   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
										   name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(tensor)
						s_tensor = convshared(tensor)
						s_ns = [TensorSelectSigmoid(shared_axes=[2, 3])([ns_tensor, s_tensor])]
						res += [s_ns]
				x = res
			elif component == 'convsnsdyrec':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]

				else:
					res = [ConvBank(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									kernel_regularizer=w_reg, activation=activation, name='ConvBank{}'.format(block_index), use_bias=use_bias)(x)]
				x = res
			elif component == 'convsnsdyrecv2':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]

				else:
					res = [ConvBankv2(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding,
									  kernel_initializer=initializion,
									  kernel_regularizer=w_reg, activation=activation, name='ConvBank{}'.format(block_index), use_bias=use_bias)(x)]
				x = res
			elif component == 'convsnsdyrec_non_optimized':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				shratio = param['shratio']
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				convshared = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)
				if len(x) == 1:
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]

				elif len(x) == 2:
					for idx, tensor in enumerate(x):
						# res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
						#                kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx),
						# use_bias=use_bias)(
						# 	tensor)]
						ns_tensor = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding,
										   kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
										   name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(tensor)
						s_tensor = convshared(tensor)
						s_ns = [
							TensorSelectSigmoid(shared_axes=[2, 3], name='s_ns_Select_histshow{}{}'.format(block_index, idx))([ns_tensor, s_tensor])]
						res += [s_ns]
				elif len(x) == 4:
					group1 = [0, 1]
					group1_conv = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding,
										 kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
										 name=CONVSH_NAME.format(block_index) + 'Group1', use_bias=use_bias)
					group2_conv = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding,
										 kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
										 name=CONVSH_NAME.format(block_index) + 'Group2', use_bias=use_bias)
					for idx, tensor in enumerate(x):
						ns_conv = Conv2D(filters=int(nb_filter * expand_rate), kernel_size=kernel_size, padding=padding,
										 kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
										 name=CONV_NAME.format(block_index, idx), use_bias=use_bias)
						ns_tensor = ns_conv(tensor)
						s_tensor = convshared(tensor)

						if idx in group1:
							group_tensor = group1_conv(tensor)
						else:
							group_tensor = group2_conv(tensor)
						ns_g = TensorSelectSigmoid(shared_axes=[2, 3], name='ns_g_Select_histshow_ns_g{}{}'.format(block_index, idx))(
							[ns_tensor, group_tensor])
						s_ns_g = TensorSelectSigmoid(shared_axes=[2, 3], name='s_nsg_Select_histshow_s_nsg{}{}'.format(block_index, idx))(
							[ns_g, s_tensor])
						# if idx in group1:
						# 	group_conv = group1_conv
						# else:
						# 	group_conv = group2_conv
						# s_ns_g = ConvBankAgg(filter_size=nb_filter*expand_rate,conv_list_index_zero_not_shared=[ns_conv, group_conv,
						#                                                                                         convshared])(tensor)
						res += [s_ns_g]
				x = res
			elif component == 'ber':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'BER', index))(tensor)]
				x = res
			elif component == 'ger':
				x = node_list_to_list(x)
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				temp = []
				res_sign = []
				res = Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'GER', 0))(x[0])
				res_sign += [Lambda(K.sign)(res[0])]
				res_sign += [Lambda(-K.sign)(res[1])]
				for idx, tensor in enumerate(res):
					temp += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(
						tensor)]
				res = temp
				Slice
				res[0] = multiply([res[0], res_sign[0]])
				res[1] = multiply([res[1], res_sign[1]])
				x = [add(res)]
			elif component == 'ger1c':
				x = node_list_to_list(x)
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				temp = []
				res_sign = []
				res = Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'GER', 0))(x[0])
				res_sign += [Lambda(K.sign)(res[0])]
				res_sign += [Lambda(lambda x: -K.sign(x))(res[1])]
				for idx, tensor in enumerate(res):
					temp += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(
						tensor)]
				res = temp
				res_sign[0] = Slice(1)(res_sign[0])
				res_sign[1] = Slice(1)(res_sign[1])
				res[0] = multiply([res[0], res_sign[0]])
				res[1] = multiply([res[1], res_sign[1]])
				x = [add(res)]
			elif component == 'ger2c':
				x = node_list_to_list(x)
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				temp = []
				res = []
				sign_slice = Slice(1)(x[0])

				control = Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'GER', 0))(sign_slice)
				control_1 = []
				control_1 += [Lambda(K.sign)(control[0])]
				control_1 += [Lambda(lambda x: K.sign(x))(control[1])]
				res += [multiply([x[0], control_1[0]])]
				res += [multiply([x[0], control_1[1]])]
				for idx, tensor in enumerate(res):
					temp += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(
						tensor)]
				res = temp
				res[0] = multiply([res[0], control_1[0]])
				res[1] = multiply([res[1], control_1[1]])
				x = [add(res)]
			elif component == 'gertreebinary':
				def binary_block(x, idx, block_index, iteration, filter_nb=0, mask_activation='relu'):
					filter_nb_1 = x._shape_as_list()[1] if filter_nb == 0 else filter_nb
					x = Conv2D(filters=filter_nb_1, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
							   kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=use_bias)(x)
					if filter_nb_1 == 1 or iteration == 0:
						return x
					res = []
					temp = []
					sign_slice, x = Split(int(filter_nb_1 // 2))(x)
					###
					control_1 = MaskBirelu(mask_activation, name=ACT_NAME_RULE.format(block_index, 'MASK', idx))(sign_slice)
					###
					res += [multiply([x, control_1[0]])]
					res += [multiply([x, control_1[1]])]
					block_res = []
					for idx2, tensor in enumerate(res):
						block_index += 1
						block_res += [binary_block(tensor, (4 * idx) + (idx2 + 1), block_index, iteration - 1, mask_activation=mask_activation)]
					res[0] = multiply([block_res[0], control_1[0]])
					res[1] = multiply([block_res[1], control_1[1]])
					res[0] = concatenate([res[0], control_1[0]], axis=1)
					res[1] = concatenate([res[1], control_1[1]], axis=1)
					x = add(res)
					return x

				x = node_list_to_list(x)
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				iteration = int(param['iter']) if 'iter' in param else -1
				mask_activation = param['mactivation'] if 'mactivation' in param else 'relu'
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = [binary_block(x[0], idx=0, block_index=block_index, iteration=iteration, filter_nb=nb_filter, mask_activation=mask_activation)]
			elif component == 'gertree':
				def block(x, idx, block_index, iteration=1):
					filter = x._shape_as_list()[1]
					x = Conv2D(filters=filter, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion, kernel_regularizer=w_reg,
							   activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=True)(x)
					if filter == 1 or iteration == 0:
						return x
					res = []
					temp = []
					sign_slice, x = Split(1)(x)
					###
					control_1 = MaskBirelu('relu', name=ACT_NAME_RULE.format(block_index, 'MASK', idx))(sign_slice)
					###
					res += [multiply([x, control_1[0]])]
					res += [multiply([x, control_1[1]])]
					block_res = []
					for idx2, tensor in enumerate(res):
						block_index += 1
						block_res += [block(tensor, (4 * idx) + (idx2 + 1), block_index, iteration - 1)]
					res[0] = multiply([block_res[0], control_1[0]])
					res[1] = multiply([block_res[1], control_1[1]])
					res[0] = concatenate([res[0], control_1[0]], axis=1)
					res[1] = concatenate([res[1], control_1[1]], axis=1)
					x = add(res)
					return x

				x = node_list_to_list(x)
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = [block(x[0], 0, block_index, 3)]
			elif component == 'becbnr':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				y = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index),
										 use_bias=use_bias), x)
				res = []

				for idx, tensor in enumerate(y):
					res += [BatchNormalization(axis=1, epsilon=1e-5, name=BATCH_NORM_RULE.format(block_index, idx))(tensor)]
				y = res
				res = []
				for index, tensor in enumerate(y):
					res += [Activation('relu', name=ACT_NAME_RULE.format(block_index, 'RELU', index))(tensor)]
				x = res + x
			elif component == 'becbnshr':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				y = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index),
										 use_bias=use_bias), x)
				res = []
				y = Layer_on_list(BatchNormalization(axis=1, epsilon=1e-5, name=BATCH_NORMSH_RULE.format(block_index)), y)
				for index, tensor in enumerate(y):
					res += [Activation('relu', name=ACT_NAME_RULE.format(block_index, 'RELU', index))(tensor)]
				x = res + x
			elif component == 'bes':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Birelu('sigmoid', name=ACT_NAME_RULE.format(block_index, 'BER', index))(tensor)]
				x = res
			elif component == 'bertanh':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Activation('tanh', name=ACT_NAME_RULE.format(block_index, 'TANH', index))(tensor)]
				x = res
				res = []
				for index, tensor in enumerate(x):
					res += [Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'BER', index))(tensor)]
				x = res
			elif component == 'relu':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Activation('relu', name=ACT_NAME_RULE.format(block_index, 'RELU', index))(tensor)]
				x = res
				block_index +=1
			elif component == 'sigmoid':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Activation('sigmoid', name=ACT_NAME_RULE.format(block_index, 'RELU', index))(tensor)]
				x = res
			elif component == 'crelu':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Crelu('relu', name=ACT_NAME_RULE.format(block_index, 'CRELU', index))(tensor)]
				x = res
			elif component == 'averagepool':
				pool_size = int(param['r'])
				strides = int(param['s'])
				pool_size = (pool_size, pool_size)
				strides = (strides, strides)
				padding = param['pad'] if 'pad' in param.keys() else 'valid'
				x = node_list_to_list(x)
				x = Layer_on_list(
					AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, name=POOL_NAME_RULE.format(block_index, 'AVERAGE')), x)
			elif component == 'maxpool':
				pool_size = int(param['r'])
				strides = int(param['s'])
				padding = param['pad'] if 'pad' in param.keys() else 'valid'
				pool_size = (pool_size, pool_size)
				strides = (strides, strides)
				x = node_list_to_list(x)
				x = Layer_on_list(MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=POOL_NAME_RULE.format(block_index,
																																 'MAX')),
								  x)
			# elif component == 'minch':
			# 	# pool_size = int(param['r'])
			# 	# strides = int(param['s'])
			# 	# pool_size = (pool_size, pool_size)
			# 	# strides = (strides, strides)
			# 	# x = node_list_to_list(x)
			# 	# x = Layer_on_list(Merge(mode='max',concat_axis=1), x)
			elif component == 'maxch':
				x = node_list_to_list(x)
				x = Layer_on_list(Merge(mode='max', concat_axis=1), x)
			elif component == 'dropout':
				drop_rate = param['p']
				x = node_list_to_list(x)
				x = Layer_on_list(Dropout(rate=drop_rate, name='BLOCK{}_DROPOUT'.format(block_index)), x)
			elif component == 'densesh':
				n = int(param['n'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				x = node_list_to_list(x)
				x = Layer_on_list(Dense(int(n*expand_rate), kernel_initializer='he_uniform', name='BLOCK{}_DENSE'.format(block_index),
										kernel_regularizer=w_reg), x)
				block_index += 1
			elif component == 'denseshfixedfilter':
				n = int(param['n'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Dense(int(n), kernel_initializer='he_uniform', name='BLOCK{}_DENSE'.format(block_index), kernel_regularizer=w_reg),
					x)
				block_index += 1
			elif component == 'flattensh':
				x = node_list_to_list(x)
				x = Layer_on_list(Flatten(), x)
			elif component == 'softmax':
				x = node_list_to_list(x)
				activations
				x = Layer_on_list(Activation('softmax', name='SOFTMAX'), x)
			elif component == 'psoftmax':
				x = node_list_to_list(x)
				activations
				x = Layer_on_list(PSoftMax(name='Parallel_SOFTMAX'), x)
			elif component == 'merge_branch_add':
				x = node_list_to_list(x)
				if not x.__len__() == 1:
					x = [add(x)]
				else:
					warnings.warn('tensor list has one element, Merge Branch is not needed')
			elif component == 'merge_branch_average':
				x = node_list_to_list(x)
				if not x.__len__() == 1:
					x = [average(x)]
				else:
					warnings.warn('tensor list has one element, Merge Branch is not needed')
			elif component == 'max_entropy_branch_select':
				x = node_list_to_list(x)
				x = [MaxEntropy()(x)]
			elif component == 'finvan':
				x = node_list_to_list(x)
				if not x.__len__() == 1:
					raise ValueError('output node is a list of tensor, Probably forgot about merging branch')
				x = x[0]
				# return Model(input=img_input, output=x)
				return {'model': None, 'in': img_input, 'out': x}
			elif component == 'fin':
				x = node_list_to_list(x)
				# if not x.__len__() == 1:
				# 	raise ValueError('output node is a list of tensor, Probably forgot about merging branch')
				x = x[0]
				# return Model(input=img_input, output=x)
				return {'model': Model(inputs=[img_input], outputs=x), 'in': img_input, 'out': x}
			############################################ TEMP LAYERS##############################
			elif component == 'xlogx':
				x = node_list_to_list(x)

				res = []
				for index, tensor in enumerate(x):
					res += [XLogX(name=ACT_NAME_RULE.format(block_index, 'RELULogX', index))(tensor)]
				x = res
			elif component == 'llu':
				x = node_list_to_list(x)

				res = []
				for index, tensor in enumerate(x):
					res += [LLU(name=ACT_NAME_RULE.format(block_index, 'RELULogX', index))(tensor)]
				x = res
			elif component == 'relusplit':
				x = node_list_to_list(x)
				nb_child = int(param['child'])
				res = []
				for idx, tensor in enumerate(x):
					res += [ReluSplit(nb_child, name=ACT_NAME_RULE.format(block_index, 'SplitRELU', idx))(tensor)]
				x = res
			elif component == 'relusplitsample':
				x = node_list_to_list(x)
				nb_child = int(param['child'])
				res = []
				for idx, tensor in enumerate(x):
					res += [ReluSplit(nb_child, name=ACT_NAME_RULE.format(block_index, 'SplitRELU', idx))(tensor)]
				x = res
			elif component == 'ampconv':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					res += [AmpConv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									  kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for idx, tensor in enumerate(x):
						res += [AmpConv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										  kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx),
										  use_bias=use_bias)(tensor)]
				x = res
			elif component == 'splitconv':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					res += [AmpConv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									  kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias)(x[0])]
				else:
					for idx, tensor in enumerate(x):
						res += [AmpConv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										  kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx),
										  use_bias=use_bias)(tensor)]
				x = res
			elif component == 'ampconv1':
				norm = float(param['norm'])
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					res += [AmpConv2Dv1(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), use_bias=use_bias,
										norm=norm)(x[0])]
				else:
					for idx, tensor in enumerate(x):
						res += [
							AmpConv2Dv1(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx), use_bias=use_bias,
										norm=norm)(tensor)]
				x = res
			elif component == 'adarelu':
				x = node_list_to_list(x)
				x = [AdaRelu(name=ACT_NAME_RULE.format(block_index, 'ADARELU', 0))(x)]
			elif component == 'amprelu':
				x = node_list_to_list(x)
				norm = float(param['norm'])
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpRelu(norm, name=ACT_NAME_RULE.format(block_index, 'AMPRELU', idx))(tensor)]
				x = res
			elif component == 'amprelu':
				x = node_list_to_list(x)
				norm = float(param['norm'])
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpRelu(norm, name=ACT_NAME_RULE.format(block_index, 'AMPRELU', idx))(tensor)]
				x = res
			elif component == 'ampbmeanrelu':
				x = node_list_to_list(x)
				norm = float(param['norm'])
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpBiMeanRelu(norm, name=ACT_NAME_RULE.format(block_index, 'AMPRELU', idx))(tensor)]
				x = res
			elif component == 'ampreluc':
				x = node_list_to_list(x)
				norm = float(param['norm'])
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpRelu(norm, name=ACT_NAME_RULE.format(block_index, 'AMPRELU', idx))(tensor)]
				x = res
			elif component == 'ampreluch':
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpReluch(name=ACT_NAME_RULE.format(block_index, 'AMPRELU', idx))(tensor)]
				x = res
			elif component == 'ampber':
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpBER(name=ACT_NAME_RULE.format(block_index, 'AMPBER', idx))(tensor)]
				x = res
			elif component == 'ampber1':
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					res += [AmpBER1(name=ACT_NAME_RULE.format(block_index, 'AMPBER1', idx))(tensor)]
				x = res
			elif component == 'convshfixed':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
										 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), trainable=False,
										 use_bias=use_bias), x)
			elif component == 'convtansh':
				block_index += 1
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				kernel_max_init = float(param['k_max'])
				bias_max_init = float(param['b_max']) if 'b_max' in param else 1.0
				w_reg = l1_l2_tanh(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Conv2DTanh(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
							   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index),
							   kernel_max_init=kernel_max_init,
							   bias_max_init=bias_max_init, use_bias=use_bias), x)
			elif component == 'convcolsh':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				col = int(param['col'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				res = []
				convcol_list = []
				for conv_idx in np.arange(col):
					convcol_list += [
						Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
							   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format('{}_COL_{}'.format(block_index, conv_idx)),
							   use_bias=use_bias)]
				if not type(x[0]) is list:
					col_compatible_x = []
					for branch in x:
						col_compatible_x += [[branch]]
					x = col_compatible_x
				for branch in x:
					branch_res = []
					for tensor in branch:
						for conv_col in convcol_list:
							branch_res += [conv_col(tensor)]
					res += [branch_res]
				x = res
			elif component == 'convyy':
				block_index += 1
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				nb_filter = int(param['f'])
				yang_sel_prob = int(param['yp'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Conv2DYingYang(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
								   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), ying_yang=True,
								   use_bias=use_bias, yangsel=yang_sel_prob), x)
			elif component == 'convyyaux':
				block_index += 1
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				nb_filter = int(param['f'])
				yang_sel_prob = float(param['yp'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				yang_w_reg = l1_l2(l1=w_reg_l1_val * 2, l2=w_reg_l2_val * 2)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Conv2DYingYangAux(filters=nb_filter * expand_rate, kernel_size=kernel_size, yang_aux_phase=yang_aux_phase, padding=padding, \
									  kernel_initializer=initializion, kernel_regularizer=w_reg, activation=activation,
									  name=CONVSH_NAME.format(block_index), ying_yang=True, use_bias=use_bias, yang_sel=yang_sel_prob,
									  yang_w_reg=yang_w_reg), x)
			elif component == 'convyang':
				block_index += 1
				use_bias = bool(param['bias'] if 'bias' in param else 1)
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Conv2dYang(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
							   kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), ying_yang=True,
							   use_bias=use_bias), x)
			elif component == 'convyy_rand':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if 'l1_val' in param else 0
				w_reg_l2_val = param['l2_val'] if 'l2_val' in param else 0
				padding = param['padding'] if 'padding' in param else 'same'
				activation = param['activation'] if 'activation' in param else None
				initializion = param['int'] if 'init' in param else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(
					Conv2DRandomYang(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
									 kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index), ying_yang=True,
									 use_bias=use_bias), x)
			elif component == 'noise_mul':
				drop_rate = param['p']
				axes = [[2, 3], [1, 2, 3]]
				channel = int(param['chw'])
				x = node_list_to_list(x)
				x = Layer_on_list(MulNoise(p=drop_rate, shared_axes=axes[channel], name='BLOCK{}_DROPOUT'.format(block_index)), x)
			elif component == 'relucol':
				res = []

				for index, branch in enumerate(x):
					pos_list = []
					for col_idx, tensor in enumerate(branch):
						branch_res = Activation('relu', name=ACT_NAME_RULE.format(block_index, 'BER', '{}_COL_{}'.format(index, col_idx)))(tensor)
						pos_list += [branch_res]
					res += [pos_list]
				x = res
			elif component == 'bercol':
				# ber compatible wih convcol
				res = []

				for index, branch in enumerate(x):
					pos_list = []
					neg_list = []
					for col_idx, tensor in enumerate(branch):
						branch_res = Birelu('relu', name=ACT_NAME_RULE.format(block_index, 'BER', '{}_COL_{}'.format(index, col_idx)))(tensor)
						pos_list += [branch_res[0]]
						neg_list += [branch_res[1]]
					res += [pos_list, neg_list]
				x = res
			elif component == 'concat_col':
				res = []

				for branch in x:
					if not branch.__len__() == 1:
						res += [[concatenate(branch, axis=1)]]
					else:
						res += [branch]
						warnings.warn('tensor list has one element, Merge Branch is not needed')
				x = res
			elif component == 'ifc':
				out_len = int(param['out'])
				x = node_list_to_list(x)
				ifc_out = [FullyConnectedTensors(output_tensors_len=out_len, shared_axes=[2, 3], name='IFC_histshow{}'.format(block_index))(x)]
				ifc_out = node_list_to_list(ifc_out)
				res = []
				for tensor in ifc_out:
					res += [[tensor]]
				x = res
			elif component == 'ifcv2':
				out_len = int(param['out'])
				x = node_list_to_list(x)
				ifc_out = [FullyConnectedTensorsv2(output_tensors_len=out_len, shared_axes=[2, 3], name='IFC_histshow{}'.format(block_index))(x)]
				ifc_out = node_list_to_list(ifc_out)
				res = []
				for tensor in ifc_out:
					res += [[tensor]]
				x = res
			elif component == 'iadd':
				x = node_list_to_list(x)
				x = [add(x)]
			elif component == 'imean':
				x = node_list_to_list(x)
				x = [average(x)]
			elif component == 'mask':
				x = node_list_to_list(x)
				# x = Layer_on_list(Activation('hard_sigmoid'),x)
				x_orig = queue_dict[param['name']]
				res = []
				for tensor in x:
					res += [add([tensor, x_orig[0]])]
				x = res
			elif component == 'ifconcat':
				x = node_list_to_list(x)
				if len(x) == 1:
					continue
				x = [concatenate(x, axis=1)]
			elif component == 'ifctan':
				out_len = int(param['out'])
				x = node_list_to_list(x)
				ifc_out = [FullyConnectedTensorsTanh(output_tensors_len=out_len, shared_axes=[2, 3])(x)]
				ifc_out = node_list_to_list(ifc_out)
				res = []
				for tensor in ifc_out:
					res += [[tensor]]
				x = res
			elif component == 'concat':
				x = node_list_to_list(x)
				if not x.__len__() == 1:
					x = [concatenate(x, axis=1)]
				else:
					warnings.warn('tensor list has one element, Merge Branch is not needed')
			elif component == 'maxav':
				pool_size = int(param['r'])
				strides = int(param['s'])
				pool_size = (pool_size, pool_size)
				strides = (strides, strides)
				x = node_list_to_list(x)
				x_max = Layer_on_list(MaxPooling2D(pool_size=pool_size, strides=strides, name=POOL_NAME_RULE.format(block_index, 'MAX')), x)
				x_av = Layer_on_list(AveragePooling2D(pool_size=pool_size, strides=strides, name=POOL_NAME_RULE.format(block_index, 'AVERAGE')), x)
				x = x_max + x_av
			elif component == 'maxaverage_entropy_select':
				# selects  max entropy and combines it with the average of all tensors. must be after softmax.
				average_rate = float(param['average_rate'])
				if average_rate > 1:
					raise ValueError('Average rate should be less than 1')
				x = node_list_to_list(x)
				max_x = MaxEntropy()(x)
				average_x = average(x)
				weighted_average = Lambda(lambda x: x * average_rate)(average_x)
				# weighted_average = multiply([(average_rate),average(x)])
				weighte_max = Lambda(lambda x: x * (1 - average_rate))(max_x)
				x = [add([weighted_average, weighte_max])]
			# x = add([(1 - average_rate) * MaxEntropy(x), (average_rate * average_vote)])
			elif component == 'weighted_softmax':
				# calculates of softmax(x) and and takes averages based on -entropy(softmax(x)) for weights. weights = softmax(-etropy(softmax(
				# x_branch)))
				x = node_list_to_list(x)
				x = [WeightedAverageSoftMax(name='SOFTMAX_Weighted')(x)]
			elif component == 'weighted_average_pred':
				# Calculates weighted average of predictions based on softmax of entropies. weights are max_entropy- entropy
				x = node_list_to_list(x)
				x = [WeightedAverageWithEntropy()(x)]
			elif component == 'weighted_average_pred_1':
				# Calculates weighted average of predictions based on softmax of entropies. weights are max_entropy- entropy
				x = node_list_to_list(x)
				x = [WeightedAverageWithEntropy0Max(name='SOFTMAX_Weighted')(x)]
			elif component == 'softmax_activity_reg':
				# Has activity regularizer tries to maximize entropy of those having high entropy. should be after softmax
				x = node_list_to_list(x)
				x = [SoftmaxEntropyActivityRegLayer(name='activity_reg')(x)]
			elif component == 'berp':
				# #the output tensors would be 2^(n+C) if we have n input tensors and C as kernel channels
				random_permute_flag = int(param['random_permute'])
				prob = float(param['p']) if random_permute_flag == 1 else 0

				max_perm = int(param['max_perm'])
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					ber_tensor_list = Birelu('relu', name='ACT_BER_L' + str(block_index) + 'I_' + str(idx))(tensor)
					res += [PermuteChannels(max_perm=max_perm, random_permute=random_permute_flag, p=prob,
											name=ACT_NAME_RULE.format(block_index, 'BERP', idx))(ber_tensor_list)]
				x = res
			elif component == 'random_average':
				# #the output tensors would be 2^(n+C) if we have n input tensors and C as kernel channels
				x = node_list_to_list(x)
				x = [RandomAveragePooling2D((32, 32), padding='same', strides=1)(x[0])]
			elif component == 'biperm':
				# #the output tensors would be 2^(n+C) if we have n input tensors and C as kernel channels
				random_permute_flag = int(param['random_permute'])
				prob = float(param['p']) if random_permute_flag == 1 else 0

				max_perm = int(param['max_perm'])
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					ber_tensor_list = Birelu('relu', name='ACT_BER_L' + str(block_index) + 'I_' + str(idx))(tensor)
					res += [BiPermuteChannels(max_perm=max_perm, random_permute=random_permute_flag, p=prob,
											  name=ACT_NAME_RULE.format(block_index, 'BIBERP', idx))(ber_tensor_list)]
				x = res

			else:
				raise ValueError(component + ' Not Found')


def node_list_to_list(array_tensor):
	'convert a hiearchial list to flat list'
	result = []
	if not type(array_tensor) == list:
		return array_tensor
	else:
		for tensor_list in array_tensor:
			a = node_list_to_list(tensor_list)
			if type(a) == list:
				result += a
			else:
				result += [a]
	return result


def parse_model_string(model_string):
	model_string_list = model_string.split('->')
	model_filter_list = []
	nb_filter_list = []
	filter_size_list = []
	expand_dropout = 1
	for block in model_string_list:
		filter_dict = {}
		filter = block.split('|')
		filter_name = filter[0]
		filter_dict['type'] = filter_name
		filter_dict['param'] = {}
		if filter.__len__() == 1:
			model_filter_list += [filter_dict]
			continue
		parameters = filter[1]
		parameters = parameters.split(',')
		for parameter in parameters:
			param = parameter.split(':')
			param_name = param[0]
			param_val = param[1]
			if not str(param_val).isalpha():
				filter_dict['param'][param_name] = float(param_val)
				if param_name == 'r':
					filter_size_list += [int(param_val)]
				if param_name == 'f':
					nb_filter_list += [int(param_val)]
				if param_name == 'p':
					expand_dropout = float(param_val)
			else:
				filter_dict['param'][param_name] = param_val
		model_filter_list += [filter_dict]

	return {
		'filters'       : model_filter_list,
		'r_field_size'  : filter_size_list,
		'conv_nb_filter': nb_filter_list,
		'ex_drop'       : expand_dropout,
		'dict'          : model_filter_list
	}


def get_model(opts, model_string):
	model_dict = parse_model_string(model_string)
	model_output_dict = model_constructor(opts=opts, model_dict=model_dict['dict'])
	return model_output_dict


def get_model_out_dict(opts, model_string):
	model_dict = parse_model_string(model_string)
	model_output_dict = model_constructor(opts=opts, model_dict=model_dict['dict'])
	return model_output_dict


if __name__ == '__main__':
	model_string = 'rbe|f:32,r:5,p:.75' \
				   '->rbe|f:64,r:5' \
				   '->rbe|f:128,r:5' \
				   '->s|f:128,r:5' \
				   '->mp|s:2,r:3' \
				   '->s|f:256,r:3' \
				   '->ap|s:2,r:3' \
				   '->s|f:512,r:5' \
				   '->ap|s:2,r:3' \
				   '->s|f:256,r:4' \
				   '->ap|s:2,r:3'
	x = parse_model_string(model_string)
	print(x)
