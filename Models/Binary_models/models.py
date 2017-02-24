from Models.binary_net import gate_layer_on_list
from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D, Convolution2D, \
	merge
# from Layers.gate_layer import gate_layer, gate_layer_on_list, maxpool_on_list, averagepool_on_list
from Layers.layer_wrappers.on_list_wrappers import conv_birelu_expand_on_list,conv_birelu_merge_on_list,\
	conv_birelu_swap_on_list,max_pool_on_list,avg_pool_on_list
from utils.opt_utils import get_filter_size,get_gate_activation
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
def model_constructor(layer_sequence,opts,nb_classes,input_shape):
	img_input = Input(shape=input_shape, name='image_batch')
	x = [img_input]
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	layer_index_t = 0
	f_size = [3]
	nb_filter = 32
	for component in layer_sequence:
		if not layer_index_t == 0:
			input_shape = (None,None,None)
		if component=='exp':

			x = conv_birelu_expand_on_list(input_tensor_list=x,nb_filter=int(nb_filter * expand_rate), filter_size=f_size[0],
	                               input_shape=input_shape, w_reg=None,
	                       gate_activation=get_gate_activation(opts), layer_index=layer_index_t,border_mode='same')
			nb_filter = nb_filter / 2
		if component=='swap':
			x = conv_birelu_swap_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate), filter_size=f_size[0],
			                               input_shape=input_shape, w_reg=None,
			                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                               border_mode='same')
		if component=='rmerge':
			nb_filter=nb_filter*2
			x = conv_birelu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate), filter_size=f_size[0],
			                               input_shape=input_shape, w_reg=None,
			                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			                               border_mode='same')
		if component=='avg':
			x =avg_pool_on_list(input_tensor_list=x,strides=(2,2),layer_index=layer_index_t)
		if component == 'max':
			x = max_pool_on_list(input_tensor_list=x, strides=(2, 2), layer_index=layer_index_t)
		layer_index_t+=1
	merged = x[0]
	if not x.__len__() == 1:
		for input1 in x[1:]:
			merged = merge([merged, input1], mode='concat', concat_axis=1)
	x = Flatten(name='flatten')(merged)
	x = Dense(nb_classes)(x)
	x = Activation('softmax')(x)
	model = Model(input=img_input, output=x)
	return model
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
def get_model(opts, input_shape, nb_classes,model_string):
	model_string_list = model_string.split('-')
	return model_constructor(model_string_list,opts=opts,nb_classes=nb_classes,input_shape=input_shape)