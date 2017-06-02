from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Conv2D,SpatialDropout2D

from Layers.layer_wrappers.on_list_wrappers import *
from utils.opt_utils import get_filter_size,get_gate_activation
from keras.regularizers import l1,l2,l1_l2
from keras.layers.merge import add,concatenate,average
import numpy as np
layer_index=0
def Layer_on_list(layer, tensor_list):
	res = []
	tensor_list = node_list_to_list(tensor_list)
	for x in tensor_list:
		res+=[layer(x)]
	return res
def get_layer_index():
	global layer_index
	layer_index=layer_index+1
	return layer_index
def model_constructor(layer_sequence,opts,nb_classes,input_shape,nb_filter_list=None,filter_size_list = None,
                      model_dict=None):
	'nb_filter_list is total filters used in each layer.filter size is for convolution'

	img_input = Input(shape=input_shape, name='image_batch')
	print K.floatx()
	x = [img_input]
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	layer_index_t = 0
	filter_size_index =0
	conv_nb_filterindex=0
	branch = 1
	batch_norm = False
	fully_drop =0
	leak_rate=0
	child_probability=.5
	counter = False # for prelu permutation
	flatten_flag = False
	no_class_dense=False
	for layer in model_dict:
		w_regularizer_str = opts['model_opts']['param_dict']['w_regularizer']['method']
		if w_regularizer_str == 'l1':
			w_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
			b_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
		if w_regularizer_str == None:
			w_reg = None
			b_reg = None
		if w_regularizer_str == 'l2':
			w_reg = l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
			b_reg = l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
		component = layer['type']

		if not layer_index_t == 0:
			input_shape = (None,None,None)
		# if nb_filter_list is not None and component in ['e','s','rm','am','rbe','rbs']:
		# 	nb_filter=nb_filter_list[conv_nb_filterindex]
		# if filter_size_list is not None:
		# 	f_size = filter_size_list[filter_size_index]
		param =layer['param']
		with tf.name_scope(component) as scope:
			if component == 'convsh':
				nb_filter = int(param['f'])
				f_size = int(param['r'])
				w_reg_l1_val = param['l1_val'] if param.has_key('l1_val') else 0
				w_reg_l2_val = param['l2_val'] if param.has_key('l2_val') else 0
				padding = param['padding'] if param.has_key('padding') else 'same'
				activation = param['activation'] if param.has_key('activation') else None
				initializion = param['int'] if param.has_key('init') else 'he_normal'
				w_reg = l1_l2(l1=w_reg_l1_val,l2=w_reg_l2_val)
				kernel_size = (f_size,f_size)
				x = node_list_to_list(x)
				x = Layer_on_list(Conv2D(filters=nb_filter,kernel_size=kernel_size,padding=padding,
				                         kernel_initializer=initializion,kernel_regularizer=w_reg,activation=activation,
				                         name='CONV_'+str(layer_index_t) ),x)
			elif component == 'ber':
				x = node_list_to_list(x)
				res = []
				for index,tensor in enumerate(x):
					res += [Birelu('relu', name='ACT_BER_L' + str(layer_index_t)+'I_'+str(index))(tensor)]
				x = res
			elif component == 'relu':
				x = node_list_to_list(x)
				res = []
				for index,tensor in enumerate(x):
					res+=[Activation('relu',name='ACT_RELU_L' + str(layer_index_t)+'I_'+str(index))(tensor)]
				x = res
			elif component=='crelu':
				x = node_list_to_list(x)
				res = []
				for index, tensor in enumerate(x):
					res += [Crelu('relu', name='ACT_CRELU_L' + str(layer_index_t) + 'I_' + str(index))(tensor)]
				x = res
			elif component == 'averagepool':
				pool_size = int(param['r'])
				strides = int(param['s'])
				pool_size = (pool_size,pool_size)
				strides = (strides, strides)
				x = node_list_to_list(x)
				x = Layer_on_list(AveragePooling2D(pool_size=pool_size,strides=strides, name='POOL_AVERAGE_' + str(
					layer_index_t)), x)
			elif component == 'maxpool':
				pool_size = int(param['r'])
				strides = int(param['s'])
				pool_size = (pool_size, pool_size)
				strides = (strides,strides)
				x = node_list_to_list(x)
				x = Layer_on_list(MaxPooling2D(pool_size=pool_size,strides=strides, name='POOL_MAX_' + str(
					layer_index_t)), x)
			elif component == 'dropout':
				drop_rate = param['p']
				x = node_list_to_list(x)
				x = Layer_on_list(Dropout(rate=drop_rate,name='Dropout_' + str(layer_index_t)),x)
			elif component == 'densesh':
				n = int(param['n'])
				x = node_list_to_list(x)
				x = Layer_on_list(Dense(n,kernel_initializer='he_uneliform'),x)
			elif component=='flattensh':
				x = node_list_to_list(x)
				x = Layer_on_list(Flatten(), x)
			elif component == 'softmax':
				x = node_list_to_list(x)
				x = Layer_on_list(Activation('softmax'),x)
			elif component == 'merge_branch_add':
				x = node_list_to_list(x)
				if not x.__len__()==1:
					x = [add(x)]
				else:
					raise ValueError('Merge Branch ADD tried to merge but list has only one element')
			elif component == 'merge_branch_average':
				x = node_list_to_list(x)
				if not x.__len__() == 1:
					x = [average(x)]
				else:
					raise ValueError('Merge Branch Average tried to merge but list has only one element')
			elif component == 'fin':
				x = node_list_to_list(x)
				if not x.__len__()==1:
					raise ValueError('output node is a list of tensor, Probably forgot about merging branch')
				x = x[0]
				return Model(input=img_input, output=x)
			else:
				raise ValueError(component+' Not Found')

			layer_index_t+=1
			conv_nb_filterindex+=1
			filter_size_index+=1

	if not flatten_flag:
		if type(x)==list and (type(x[0]) == list or not x.__len__()==1) :
			x = node_list_to_list(x)
			merged =concatenate(x, axis=1)
		else:
			if type(x)==list:
				x = x[0]
			merged = x
		with tf.name_scope('Flatten'):
			x = Flatten(name='flatten')(merged)
		with tf.name_scope('Dropout'):
			if not fully_drop==0:
				x = Dropout(fully_drop)(x)
		with tf.name_scope('Dense'):
			if not no_class_dense:
				x = Dense(int(nb_classes))(x)
		with tf.name_scope('SoftMax'):
			x = Activation('softmax')(x)
	model = Model(input=img_input, output=x)
	return model
def remove_pooling_from_string(model_string):
	model_string.replace('-ap')

def node_list_to_list(array_tensor):
	'convert a hiearchial list to flat list'
	result =[]
	if not type(array_tensor)==list:
		return array_tensor
	else:
		for tensor_list in array_tensor:
			a = node_list_to_list(tensor_list)
			if type(a)==list:
				result+=a
			else:
				result+=[a]
	return result
def parse_model_string(model_string):
	model_string_list = model_string.split('->')
	model_filter_list = []
	nb_filter_list = []
	filter_size_list = []
	expand_dropout= 1
	dict = {}
	result_list = []
	for block in model_string_list:
		filter_dict = {}
		filter = block.split('|')
		filter_name = filter[0]
		filter_dict['type'] = filter_name
		filter_dict['param'] = {}
		if filter.__len__()==1:
			model_filter_list += [filter_dict]
			continue
		parameters = filter[1]
		parameters = parameters.split(',')
		for parameter in parameters:
			param = parameter.split(':')
			param_name = param[0]
			param_val = param[1]
			if not str(param_val).isalpha():
				filter_dict['param'][param_name]=float(param_val)
				if param_name == 'r':
					filter_size_list += [int(param_val)]
				if param_name == 'f':
					nb_filter_list += [int(param_val)]
				if param_name == 'p':
					expand_dropout=float(param_val)
			else:
				filter_dict['param'][param_name] = param_val
		model_filter_list+=[filter_dict]

	return {'filters':model_filter_list,'r_field_size':filter_size_list,'conv_nb_filter':nb_filter_list,
	        'ex_drop':expand_dropout,'dict':model_filter_list}
def get_model(opts, input_shape, nb_classes,model_string,nb_filter_list=None,conv_filter_size_list=None):
	model_dict = parse_model_string(model_string)

	return model_constructor(model_dict['filters'],opts=opts,nb_classes=nb_classes,input_shape=input_shape,
	                         nb_filter_list=model_dict['conv_nb_filter'],filter_size_list=model_dict['r_field_size'],
	                         model_dict = model_dict['dict'])
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