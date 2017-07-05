import warnings

from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Conv2D, Activation, AveragePooling2D, Lambda, MaxPooling2D
from keras.layers.merge import add, average
from keras.regularizers import l1_l2

from utils.modelutils.layers.binary_layers.birelu import *

layer_index = 0
# use of | will make the parser for load weight not considering string past '|'
CONVSH_NAME = 'BLOCK{}_CONV'
CONV_NAME = 'BLOCK{}_CONV-{}'
ACT_NAME_RULE = 'BLOCK{}_ACT_{}{}'
POOL_NAME_RULE = 'BLOCK{}_POOL_{}'

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
	x = [img_input]
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	block_index = 0
	for layer in model_dict:
		component = layer['type']

		param = layer['param']
		with tf.name_scope(component):
			if component == 'convsh':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if param.has_key('l1_val') else 0
				w_reg_l2_val = param['l2_val'] if param.has_key('l2_val') else 0
				padding = param['padding'] if param.has_key('padding') else 'same'
				activation = param['activation'] if param.has_key('activation') else None
				initializion = param['int'] if param.has_key('init') else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
				                         kernel_regularizer=w_reg, activation=activation, name=CONVSH_NAME.format(block_index)), x)

			elif component == 'conv':
				block_index += 1
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if param.has_key('l1_val') else 0
				w_reg_l2_val = param['l2_val'] if param.has_key('l2_val') else 0
				padding = param['padding'] if param.has_key('padding') else 'same'
				activation = param['activation'] if param.has_key('activation') else None
				initializion = param['int'] if param.has_key('init') else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val, l2=w_reg_l2_val)
				kernel_size = (f_size, f_size)
				x = node_list_to_list(x)
				res = []
				if len(x) == 1:
					raise ValueError('one branch for convolution use convsh instead for load weight compatibility')
				for idx, tensor in enumerate(x):
					res += [Conv2D(filters=nb_filter * expand_rate, kernel_size=kernel_size, padding=padding, kernel_initializer=initializion,
					               kernel_regularizer=w_reg, activation=activation, name=CONV_NAME.format(block_index, idx))(tensor)]
				x = res

			elif component == 'ber':
				x = node_list_to_list(x)
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
				x = node_list_to_list(x)
				x = Layer_on_list(AveragePooling2D(pool_size=pool_size, strides=strides, name=POOL_NAME_RULE.format(block_index, 'AVERAGE')), x)
			elif component == 'maxpool':
				pool_size = int(param['r'])
				strides = int(param['s'])
				pool_size = (pool_size, pool_size)
				strides = (strides, strides)
				x = node_list_to_list(x)
				x = Layer_on_list(MaxPooling2D(pool_size=pool_size, strides=strides, name=POOL_NAME_RULE.format(block_index, 'MAX')), x)
			elif component == 'dropout':
				drop_rate = param['p']
				x = node_list_to_list(x)
				x = Layer_on_list(Dropout(rate=drop_rate, name='BLOCK{}_DROPOUT'.format(block_index)), x)
			elif component == 'densesh':
				n = int(param['n'])
				x = node_list_to_list(x)
				x = Layer_on_list(Dense(n, kernel_initializer='he_uniform', name='BLOCK{}_DENSE'.format(block_index)), x)
				block_index += 1
			elif component == 'flattensh':
				x = node_list_to_list(x)
				x = Layer_on_list(Flatten(), x)
			elif component == 'softmax':
				x = node_list_to_list(x)
				x = Layer_on_list(Activation('softmax', name='SOFTMAX'), x)
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

			elif component == 'fin':
				x = node_list_to_list(x)
				if not x.__len__() == 1:
					raise ValueError('output node is a list of tensor, Probably forgot about merging branch')
				x = x[0]
				# return Model(input=img_input, output=x)
				return Model(inputs=[img_input],outputs=[x])


			############################################ TEMP LAYERS##############################


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
				prob = float(param['p']) if random_permute_flag==1 else 0

				max_perm = int(param['max_perm'])
				x = node_list_to_list(x)
				res = []
				for idx, tensor in enumerate(x):
					ber_tensor_list = Birelu('relu', name='ACT_BER_L' + str(block_index) + 'I_' + str(idx))(tensor)
					res += [
						PermuteChannels(max_perm=max_perm, random_permute=random_permute_flag,p=prob, name=ACT_NAME_RULE.format(block_index, 'BERP',
						                                                                                                     idx))(
							ber_tensor_list)]
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

	return model_constructor(opts=opts, model_dict=model_dict['dict'])


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
	print x
