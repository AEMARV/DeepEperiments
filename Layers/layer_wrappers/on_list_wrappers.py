from keras.layers.core import Dropout

from Layers.layer_wrappers.conv_birelu import *
from keras.layers import MaxPooling2D,AveragePooling2D,Dropout
def conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',relu_birelu_switch=1):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                                   layer_index,
                      lists_or_tensor,index=index+str(indexint),relu_birelu_switch=relu_birelu_switch)]
		else:
			result +=[conv_birelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                             index+str(indexint),
			                       layer_index,
                      lists_or_tensor,relu_birelu_switch=relu_birelu_switch)]

		indexint+=1
	return result
def conv_birelu_swap_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',relu_birelu_switch=1):
	result = []
	indexint =0

	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor[0]) == list:
			result += [
				conv_birelu_swap_on_list(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
				                            layer_index, lists_or_tensor,index=index+str(indexint),relu_birelu_switch=relu_birelu_switch)]
		else:

			result += [conv_birelu_swap(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation, index+str(indexint),
				                   layer_index, lists_or_tensor,relu_birelu_switch=relu_birelu_switch)]
		indexint+=1
	return result
def conv_relu_merge_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0'):
	result = []
	indexint =0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor[0]) == list:
			result += [
				conv_relu_merge_on_list(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
				                           layer_index, lists_or_tensor,index=index+str(indexint))]
		else:
			result += [conv_relu_merge(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation, index+str(indexint),
				                   layer_index, lists_or_tensor)]
		indexint += 1
	return result
def conv_birelu_merge_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0'):
	result = []
	indexint =0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor[0]) == list:
			result += [
				conv_birelu_merge_on_list(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
				                           layer_index, lists_or_tensor,index=index+str(indexint))]
		else:
			result += [conv_birelu_merge(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
			                            index+str(indexint),
				                   layer_index, lists_or_tensor)]
		indexint += 1
	return result
def conv_relu_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0'):
	result = []
	indexint =0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [
				conv_relu_on_list(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
				                           layer_index, lists_or_tensor,index=index+str(indexint))]
		else:
			result += [conv_relu(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
			                            index+str(indexint),
				                   layer_index, lists_or_tensor)]
		indexint += 1
	return result
def max_pool_on_list(input_tensor_list,strides,layer_index,index='0',pool_size =None):
	result = []
	indexint = 0
	if pool_size is None:
		p_size = 3
	else:
		p_size= pool_size
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [
				max_pool_on_list(lists_or_tensor,strides=strides,layer_index=layer_index, index=index + str(indexint))]
		else:
			result += [MaxPooling2D(pool_size=(p_size,p_size),strides=strides,name='Maxpool_stride'+str(
				strides)+'_psize-'+str(p_size)+'_layer'+str(layer_index)+'index'+index+str(indexint))(lists_or_tensor)]
		indexint += 1
	return result
def avg_pool_on_list(input_tensor_list,strides,layer_index,index='0',pool_size=None):
	result = []
	indexint = 0
	if pool_size is None:
		p_size = 3
	else:
		p_size= pool_size
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [avg_pool_on_list(lists_or_tensor, strides=strides, layer_index=layer_index,
			                            index=index + str(indexint))]
		else:
			result += [AveragePooling2D(pool_size=(p_size,p_size),strides=strides, name='AvgPool_stride' + str(strides) + \
			                                                                 '_psize-'+str(p_size)+'_layer' + str(
				layer_index) + '_index' + index + str(indexint))(lists_or_tensor)]
		indexint += 1
	return result
def dropout_on_list(input_tensor_list,p,layer_index,index='0'):
	result = []
	indexint = 0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [dropout_on_list(lists_or_tensor,p,layer_index=layer_index,
			                            index=index + str(indexint))]
		else:
			result += [Dropout(p=p,
			                            name='Dropout_layer' +
			                                 str(
				                            layer_index) + '_index' + index + str(indexint))(lists_or_tensor)]
		indexint += 1
	return result
def concat_on_list(input_tensor_list,n,layer_index,index = '0'):
	result = []
	indexint = 0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor[0]) == list:
			result += [
				concat_on_list(lists_or_tensor,n, layer_index, index=index + str(indexint))]
		else:
			result += [concat(input_tensor_list=lists_or_tensor,
			                             index =index + str(indexint), layer_index=layer_index)]
		indexint += 1
	return result