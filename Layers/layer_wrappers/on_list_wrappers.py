from Layers.layer_wrappers.conv_birelu import *
from keras.layers import MaxPooling2D,AveragePooling2D
def conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0'):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                                   layer_index,
                      lists_or_tensor,index=index+str(indexint))]
		else:
			result +=[conv_birelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                             index+str(indexint),
			                       layer_index,
                      lists_or_tensor)]

		indexint+=1
	return result
def conv_birelu_swap_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0'):
	result = []
	indexint =0

	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor[0]) == list:
			result += [
				conv_birelu_swap_on_list(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
				                            layer_index, lists_or_tensor,index=index+str(indexint))]
		else:

			result += [conv_birelu_swap(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation, index+str(indexint),
				                   layer_index, lists_or_tensor)]
		indexint+=1
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
			result += [conv_relu_merge(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation, index+str(indexint),
				                   layer_index, lists_or_tensor)]
		indexint += 1
	return result
def max_pool_on_list(input_tensor_list,strides,layer_index,index='0'):
	result = []
	indexint = 0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [
				max_pool_on_list(lists_or_tensor,strides=strides,layer_index=layer_index, index=index + str(indexint))]
		else:
			result += [MaxPooling2D(strides=strides,name='Maxpool_stride'+str(
				strides)+'layer'+str(layer_index)+'index'+index+str(indexint))(lists_or_tensor)]
		indexint += 1
	return result
def avg_pool_on_list(input_tensor_list,strides,layer_index,index='0'):
	result = []
	indexint = 0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [avg_pool_on_list(lists_or_tensor, strides=strides, layer_index=layer_index,
			                            index=index + str(indexint))]
		else:
			result += [AveragePooling2D(strides=strides, name='AvgPool_stride' + str(strides) + 'layer' + str(
				layer_index) + 'index' + index + str(indexint))(lists_or_tensor)]
		indexint += 1
	return result