from Models.binary_net import gate_layer_on_list
from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D, Convolution2D, \
	merge
from Layers.layer_wrappers.on_list_wrappers import conv_birelu_expand_on_list,conv_relu_merge_on_list,\
	conv_birelu_swap_on_list,max_pool_on_list,avg_pool_on_list,conv_birelu_merge_on_list,conv_relu_on_list
from utils.opt_utils import get_filter_size,get_gate_activation
from keras.utils.generic_utils import get_from_module
import numpy as np
# def model_e_0(opts, input_shape, nb_classes, input_tensor=None, include_top=True, initialization='glorot_normal'):
# 	if include_top:
# 		input_shape = input_shape
# 	else:
# 		input_shape = (3, None, None)
# 	if input_tensor is None:
# 		img_input = Input(shape=input_shape, name='image_batch')
# 	else:
# 		img_input = input_tensor
#
# 	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
# 	filter_size = get_filter_size(opts)
# 	if filter_size == -1:
# 		f_size = [3, 5, 5, 4]
# 	else:
# 		f_size = np.min([[32, 16, 7, 3], [filter_size, (filter_size + 1) / 2, filter_size, filter_size - 1]], 0)
# 	# Layer 1 Conv 5x5 32ch  border 'same' Max Pooling 3x3 stride 2 border valid
# 	x = gate_layer_on_list([img_input], int(32 * expand_rate), f_size[0], input_shape=input_shape, opts=opts,
# 	                       border_mode='same', merge_flag=False, layer_index=0)
# 	# TODO add stride for conv layer
# 	x = maxpool_on_list(x, pool_size=(3, 3), strides=(2, 2),layer_index=1, border_mode='same')
# 	#                           Layer 2 Conv 5x5 64ch  border 'same' AveragePooling 3x3 stride 2
# 	x = gate_layer_on_list(x, int(64 * expand_rate / 2), f_size[1],
# 	                       input_shape=(32, (input_shape[1] - 2), (input_shape[2]) - 2), opts=opts,
# 	                       border_mode='same', merge_flag=False, layer_index=2)
# 	x = averagepool_on_list(x, pool_size=(3, 3), strides=(2, 2),layer_index=3)
#
# 	#                           Layer 3 Conv 5x5 128ch  border 'same' AveragePooling3x3 stride 2
# 	x = gate_layer_on_list(x, int(128 * expand_rate / 4), f_size[2],
# 	                       input_shape=(32, (input_shape[1] - 2) / 2, (input_shape[2] - 2) / 2), opts=opts,
# 	                       border_mode='same', merge_flag=False,layer_index=4)
# 	x = averagepool_on_list(x, pool_size=(3, 3), strides=(2, 2), border_mode='same',layer_index=5)
# 	#                           Layer 4 Conv 4x4 64ch  border 'same' no pooling
# 	x = gate_layer_on_list(x, int(64 * expand_rate / 8), f_size[3],
# 	                       input_shape=(64, ((input_shape[1] - 2) / 2) - 2, ((input_shape[2] - 2) / 2) - 2), opts=opts,
# 	                       border_mode='valid', merge_flag=False,layer_index=6)
# 	# option 1 average pool all x
# 	# option 2 concat x list into one tensor
# 	merged = x[0]
# 	for input1 in x[1:]:
# 		merged = merge([merged, input1], mode='concat', concat_axis=1)
# 	if not include_top:
# 		model = Model(input=img_input, output=x)
# 	else:
# 		x = Flatten(name='flatten')(merged)
# 		x = Dense(nb_classes)(x)
# 		x = Activation('softmax')(x)
# 		model = Model(input=img_input, output=x)
# 	return model
layer_index=0
def get_layer_index():
	global layer_index
	layer_index=layer_index+1
	return layer_index
def model_constructor(layer_sequence,opts,nb_classes,input_shape,nb_filter_list=None,filter_size_list = None,
                      model_dict=None):
	'nb_filter_list is total filters used in each layer.filter size is for convolution'
	img_input = Input(shape=input_shape, name='image_batch')
	x = [img_input]
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	layer_index_t = 0
	f_size = 3
	nb_filter = 32
	filter_size_index =0
	conv_nb_filterindex=0
	branch = 1
	for layer in model_dict:
		component = layer['type']
		if layer['type'] not in ['e','s','rm','am','mp','ma','rbe']:
			assert 'Invalid Layer'

		if not layer_index_t == 0:
			input_shape = (None,None,None)
		# if nb_filter_list is not None and component in ['e','s','rm','am','rbe','rbs']:
		# 	nb_filter=nb_filter_list[conv_nb_filterindex]
		# if filter_size_list is not None:
		# 	f_size = filter_size_list[filter_size_index]
		param =layer['param']
		f_size = param['r']

		if component=='e':
			nb_filter = param['f']
			x = conv_birelu_expand_on_list(input_tensor_list=x,nb_filter=int(nb_filter * expand_rate/branch),
			                               filter_size=f_size,
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=layer_index_t,border_mode='same')
			branch=2*branch
		if component == 'rbe':
			if param.has_key('p'):
				dropout = param['p']
			nb_filter = param['f']
			x = conv_birelu_expand_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			                               filter_size=f_size, input_shape=input_shape, w_reg=None,
			                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                               border_mode='same',relu_birelu_switch=dropout)
			branch = 2 * branch
		if component=='s':
			nb_filter = param['f']
			x = conv_birelu_swap_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			                             filter_size=f_size,
			                               input_shape=input_shape, w_reg=None,
			                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                               border_mode='same')
		if component=='rbs':
			if param.has_key('p'):
				dropout = param['p']
			nb_filter = param['f']
			x = conv_birelu_swap_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			                             filter_size=f_size,
			                               input_shape=input_shape, w_reg=None,
			                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                               border_mode='same',relu_birelu_switch=dropout)
		if component=='rm':
			nb_filter = param['f']
			branch = branch / 2
			x = conv_relu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			                              filter_size=f_size,
			                               input_shape=input_shape, w_reg=None,
			                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                               border_mode='same')

		if component=='am':
			nb_filter = param['f']
			branch = branch / 2
			x = conv_relu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			                              filter_size=f_size, input_shape=input_shape, w_reg=None,
			                              gate_activation='avr', layer_index=layer_index_t, border_mode='same')
		if component == 'bm':
			nb_filter = param['f']
			branch = branch / 2
			x = conv_birelu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			                              filter_size=f_size, input_shape=input_shape, w_reg=None,
			                              gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                              border_mode='same')

		if component=='ap':
			x =avg_pool_on_list(input_tensor_list=x,strides=(2,2),layer_index=layer_index_t,
			                    pool_size=f_size)
			conv_nb_filterindex-=1
		if component == 'mp':
			x = max_pool_on_list(input_tensor_list=x, strides=(2, 2), layer_index=layer_index_t,pool_size=f_size)
			conv_nb_filterindex-=1
		if component == 'r':
			nb_filter = param['f']
			x = conv_relu_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			                              filter_size=f_size, input_shape=input_shape, w_reg=None,
			                              gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                              border_mode='same')

		layer_index_t+=1
		conv_nb_filterindex+=1
		filter_size_index+=1
	x = node_list_to_list(x)
	merged = x[0]
	if not x.__len__() == 1:
		for input1 in x[1:]:
			merged = merge([merged, input1], mode='concat', concat_axis=1)
	x = Flatten(name='flatten')(merged)
	x = Dense(nb_classes)(x)
	x = Activation('softmax')(x)
	model = Model(input=img_input, output=x)
	return model
def remove_pooling_from_string(model_string):
	model_string.replace('-ap')

def model_a(opts, input_shape, nb_classes, input_tensor=None, include_top=True, initialization='glorot_normal'):
	if include_top:
		input_shape = input_shape
	else:
		input_shape = (3, None, None)
	if input_tensor is None:
		img_input = Input(shape=input_shape, name='image_batch')
	else:
		img_input = input_tensor

	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	filter_size = get_filter_size(opts)
	if filter_size == -1:
		f_size = [3, 5, 5, 4]
	else:
		f_size = np.min([[32, 16, 7, 3], [filter_size, (filter_size + 1) / 2, filter_size, filter_size - 1]], 0)
	# Layer 1 Conv 5x5 32ch  border 'same' Max Pooling 3x3 stride 2 border valid
	# x = gate_layer_on_list([img_input], int(32 * expand_rate), f_size[0], input_shape=input_shape, opts=opts,
	#                        border_mode='same', merge_flag=False, layer_index=0)
	#nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      #input_tensor_list
	global layer_index
	layer_index=0
	x = conv_birelu_expand_on_list(input_tensor_list=[img_input],nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=get_layer_index(),border_mode='same')
	x = conv_birelu_expand_on_list(input_tensor_list=x,nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=get_layer_index(),border_mode='same')
	x = conv_birelu_expand_on_list(input_tensor_list=x, nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None, gate_activation=get_gate_activation(opts),
	                               layer_index=get_layer_index(), border_mode='same')
	x = conv_birelu_expand_on_list(input_tensor_list=x, nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None, gate_activation=get_gate_activation(opts),
	                               layer_index=get_layer_index(), border_mode='same')
	# x = max_pool_on_list(x, strides=(2, 2), layer_index=get_layer_index())
	x = conv_birelu_swap_on_list(input_tensor_list=x,nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=get_layer_index(),border_mode='same')
	x = conv_birelu_swap_on_list(input_tensor_list=x,nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=get_layer_index(),border_mode='same')
	x = conv_birelu_merge_on_list(input_tensor_list=x,nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=get_layer_index(),border_mode='same')
	# x = max_pool_on_list(x, strides=(2, 2), layer_index=get_layer_index())
	x = conv_birelu_merge_on_list(input_tensor_list=x,nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=get_layer_index(),border_mode='same')
	x = conv_birelu_merge_on_list(input_tensor_list=x, nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                              input_shape=input_shape, w_reg=None, gate_activation=get_gate_activation(opts),
	                              layer_index=get_layer_index(), border_mode='same')
	x = conv_birelu_merge_on_list(input_tensor_list=x, nb_filter=int(32 * expand_rate), filter_size=f_size[0],
	                              input_shape=input_shape, w_reg=None, gate_activation=get_gate_activation(opts),
	                              layer_index=get_layer_index(), border_mode='same')
	# TODO add stride for conv layer
	# x = maxpool_on_list(x, pool_size=(3, 3), strides=(2, 2),layer_index=1, border_mode='same')
	# #                           Layer 2 Conv 5x5 64ch  border 'same' AveragePooling 3x3 stride 2
	# x = gate_layer_on_list(x, int(64 * expand_rate / 2), f_size[1],
	#                        input_shape=(32, (input_shape[1] - 2), (input_shape[2]) - 2), opts=opts,
	#                        border_mode='same', merge_flag=False, layer_index=2)
	# x = averagepool_on_list(x, pool_size=(3, 3), strides=(2, 2),layer_index=3)
	#
	# #                           Layer 3 Conv 5x5 128ch  border 'same' AveragePooling3x3 stride 2
	# x = gate_layer_on_list(x, int(128 * expand_rate / 4), f_size[2],
	#                        input_shape=(32, (input_shape[1] - 2) / 2, (input_shape[2] - 2) / 2), opts=opts,
	#                        border_mode='same', merge_flag=False,layer_index=4)
	# x = averagepool_on_list(x, pool_size=(3, 3), strides=(2, 2), border_mode='same',layer_index=5)
	# #                           Layer 4 Conv 4x4 64ch  border 'same' no pooling
	# x = gate_layer_on_list(x, int(64 * expand_rate / 8), f_size[3],
	#                        input_shape=(64, ((input_shape[1] - 2) / 2) - 2, ((input_shape[2] - 2) / 2) - 2), opts=opts,
	#                        border_mode='valid', merge_flag=False,layer_index=6)
	# option 1 average pool all x
	# option 2 concat x list into one tensor
	x = node_list_to_list(x)
	merged = x[0]
	if not x.__len__()==1:
		for input1 in x[1:]:
			merged = merge([merged, input1], mode='concat', concat_axis=1)
	if not include_top:
		model = Model(input=img_input, output=x)
	else:
		x = Flatten(name='flatten')(merged)
		x = Dense(nb_classes)(x)
		x = Activation('softmax')(x)
		model = Model(input=img_input, output=x)
	return model
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
		parameters = filter[1]
		parameters = parameters.split(',')
		for parameter in parameters:
			param = parameter.split(':')
			param_name = param[0]
			param_val = param[1]
			filter_dict['param'][param_name]=float(param_val)
			if param_name == 'r':
				filter_size_list += [int(param_val)]
			if param_name == 'f':
				nb_filter_list += [int(param_val)]
			if param_name == 'p':
				expand_dropout=float(param_val)
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