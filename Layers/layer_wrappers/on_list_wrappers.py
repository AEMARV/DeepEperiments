from keras.layers.core import Dropout,Lambda
import keras.backend as K
from Layers.layer_wrappers.conv_birelu import *
from Layers.binary_layers.birelu import *
from keras.layers import MaxPooling2D,AveragePooling2D,Dropout
def conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',relu_birelu_switch=1,batch_norm=False,leak_rate =0,child_p=.5):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                                   layer_index,
                      lists_or_tensor,index=index+str(indexint),relu_birelu_switch=relu_birelu_switch,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p)]
		else:
			result +=[conv_birelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                             index+str(indexint),
			                       layer_index,
                      lists_or_tensor,relu_birelu_switch=relu_birelu_switch,batch_norm=batch_norm,
                                         leak_rate=leak_rate,child_p=child_p)]

		indexint+=1
	return result
def conv_birelu_expand_on_list_shared(conv_layer,gate_activation,layer_index,
                      input_tensor_list,index='0',drop_path_rate=1,batch_norm=False,leak_rate =0,child_p=.5
                                      ):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelu_expand_on_list_shared(conv_layer=conv_layer,
			                                         gate_activation=gate_activation,
			                                   layer_index=layer_index,
                      input_tensor_list=lists_or_tensor,index=index+str(indexint),drop_path_rate=drop_path_rate,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p)]
		else:
			result +=[conv_birelu_expand_shared(conv_layer=conv_layer,gate_activation=gate_activation,
			                             index=index+str(indexint),
			                       layer_index=layer_index,input_tensor=lists_or_tensor,
			                                    batch_norm=batch_norm,
                                         leak_rate=leak_rate,child_p=child_p,drop_path_rate=drop_path_rate)]

		indexint+=1
	return result
def conv_birelu_expand_on_list_shared_permute_channels(conv_layer,gate_activation,layer_index,
                      input_tensor_list,index='0',drop_path_rate=1,batch_norm=False,leak_rate =0,child_p=.5
                                      ,max_perm=2,random_permute_flag=0):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelu_expand_on_list_shared_permute_channels(conv_layer=conv_layer,
			                                         gate_activation=gate_activation,
			                                   layer_index=layer_index,
                      input_tensor_list=lists_or_tensor,index=index+str(indexint),drop_path_rate=drop_path_rate,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p,
                                                                        max_perm=max_perm,random_permute_flag=random_permute_flag)]
		else:
			result +=[conv_birelu_expand_shared_permute_channels(conv_layer=conv_layer,gate_activation=gate_activation,
			                             index=index+str(indexint),
			                       layer_index=layer_index,input_tensor=lists_or_tensor,
			                                    batch_norm=batch_norm,
                                         leak_rate=leak_rate,child_p=child_p,drop_path_rate=drop_path_rate,
                                                                 max_perm=max_perm,random_permute_flag=random_permute_flag)]

		indexint+=1
	return result
def conv_relu_expand_on_list_shared(conv_layer,gate_activation,layer_index,
                      input_tensor_list,index='0',drop_path_rate=1,batch_norm=False,leak_rate =0,child_p=.5
                                      ):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_relu_expand_on_list_shared(conv_layer=conv_layer,
			                                         gate_activation=gate_activation,
			                                   layer_index=layer_index,
                      input_tensor_list=lists_or_tensor,index=index+str(indexint),drop_path_rate=drop_path_rate,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p)]
		else:
			result +=[conv_relu_expand_shared(conv_layer=conv_layer,gate_activation=gate_activation,
			                             index=index+str(indexint),
			                       layer_index=layer_index,input_tensor=lists_or_tensor,
			                                    batch_norm=batch_norm,
                                         leak_rate=leak_rate,child_p=child_p,drop_path_rate=drop_path_rate)]

		indexint+=1
	return result
def conv_xavr_expand_on_list_shared(conv_layer,gate_activation,layer_index,
                      input_tensor_list,index='0',drop_path_rate=1,batch_norm=False,leak_rate =0,child_p=.5
                                      ):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_xavr_expand_on_list_shared(conv_layer=conv_layer,
			                                         gate_activation=gate_activation,
			                                   layer_index=layer_index,
                      input_tensor_list=lists_or_tensor,index=index+str(indexint),drop_path_rate=drop_path_rate,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p)]
		else:
			result +=[conv_xavr_expand_shared(conv_layer=conv_layer,
			                             index=index+str(indexint),
			                       layer_index=layer_index,input_tensor=lists_or_tensor,
			                                    batch_norm=batch_norm)]

		indexint+=1
	return result
def conv_xavrrelu_expand_on_list_shared(conv_layer,gate_activation,layer_index,
                      input_tensor_list,index='0',drop_path_rate=1,batch_norm=False,leak_rate =0,child_p=.5
                                      ):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_xavrrelu_expand_on_list_shared(conv_layer=conv_layer,
			                                         gate_activation=gate_activation,
			                                   layer_index=layer_index,
                      input_tensor_list=lists_or_tensor,index=index+str(indexint),drop_path_rate=drop_path_rate,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p)]
		else:
			result +=[conv_xavrrelu_expand_shared(conv_layer=conv_layer,
			                             index=index+str(indexint),
			                       layer_index=layer_index,input_tensor=lists_or_tensor,
			                                    batch_norm=batch_norm)]

		indexint+=1
	return result
def conv_prelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',relu_birelu_switch=1,batch_norm=False,leak_rate =0,child_p=.5,
                              prelu_counter=False):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_prelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                                   layer_index,
                      lists_or_tensor,index=index+str(indexint),relu_birelu_switch=relu_birelu_switch,
                                                batch_norm=batch_norm,leak_rate=leak_rate,child_p=child_p,
                                               prelu_counter=prelu_counter)]
		else:
			result +=[conv_prelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                             index+str(indexint),
			                       layer_index,
                      lists_or_tensor,relu_birelu_switch=relu_birelu_switch,batch_norm=batch_norm,
                                         leak_rate=leak_rate,child_p=child_p,prelu_counter=prelu_counter)]

		indexint+=1
	return result
def conv_birelu_expand_on_list_general_leak(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
                                         layer_index,
                      input_tensor_list,index='0',relu_birelu_switch=1,batch_norm=False,leak_rate =0):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelu_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                                   layer_index,
                      lists_or_tensor,index=index+str(indexint),relu_birelu_switch=relu_birelu_switch,
                                                batch_norm=batch_norm,leak_rate=0)]
		else:
			result +=[conv_birelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                             index+str(indexint),
			                       layer_index,
                      lists_or_tensor,relu_birelu_switch=relu_birelu_switch,batch_norm=batch_norm,leak_rate=0)]

		indexint+=1
	# scheme 1 : uniformly add filters. #scheme2 weighted filter.
	result_array = node_list_to_list(result)
	num_filter = int(nb_filter*leak_rate/result_array.__len__())
	if num_filter ==0:
		num_filter+=1
	tensor_donate_sliced_list = []
	for tensor_donate in result_array:
		tensor_donate_sliced = Slice(num_filter)(tensor_donate)
		tensor_donate_sliced_list+=[tensor_donate_sliced]
	tensor_donate_list = result_array
	result=donate_to_list(tensor_donate_list,result,num_filter,nb_filter,tensor_donate_sliced_list)
	return result


# Birelunary****************************************
def conv_birelunary_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',relu_birelu_switch=1):
	result = []
	indexint=0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor)==list:
			result+=[conv_birelunary_expand_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
			                                   layer_index,
                      lists_or_tensor,index=index+str(indexint),relu_birelu_switch=relu_birelu_switch)]
		else:
			result +=[conv_birelunary_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,
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
def conv_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',stride = 1,b_reg=None):
	result = []
	indexint =0
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [
				conv_on_list(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
				                           layer_index, lists_or_tensor,index=index+str(indexint))]
		else:
			result += [conv(nb_filter, filter_size, border_mode, input_shape, w_reg, gate_activation,
			                            index+str(indexint),
				                   layer_index, lists_or_tensor,stride=stride,breg=b_reg)]
		indexint += 1
	return result
def conv_relu_on_list(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,layer_index,
                      input_tensor_list,index='0',stride = 1,b_reg=None):
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
				                   layer_index, lists_or_tensor,stride=stride,breg=b_reg)]
		indexint += 1
	return result
def max_pool_on_list(input_tensor_list,strides,layer_index,index='0',pool_size =None):
	result = []
	indexint = 0
	if pool_size is None:
		p_size = int(3)
	else:
		p_size= int(pool_size)
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result += [
				max_pool_on_list(lists_or_tensor,strides=(strides),layer_index=layer_index, index=index + str(
					indexint))]
		else:
			scope = 'Maxpool_stride'+str(strides[0])+'_psize-'+str(p_size)+'_layer'+str(layer_index)+'index'+index+str(
				indexint)
			result += [MaxPooling2D(pool_size=(p_size,p_size),strides=(strides),name=scope)(lists_or_tensor)]
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
			scope = 'AvgPool_stride' + str(strides[0]) + '_psize-'+str(p_size)+'_layer' + str(layer_index) + '_index' + index + str(indexint)
			result += [AveragePooling2D(pool_size=(p_size,p_size),strides=strides, name=scope)(lists_or_tensor)]
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
# def concat_on_list2(input_tensor_list,n,layer_index,index = '0'):
# 	result = []
# 	indexint = 0
# 	for lists_or_tensor in input_tensor_list:
# 		if type(lists_or_tensor[0]) == list:
# 			result += [
# 				concat_on_list(lists_or_tensor,n, layer_index, index=index + str(indexint))]
# 		else:
# 			result += [concat(input_tensor_list=lists_or_tensor,
# 			                             index =index + str(indexint), layer_index=layer_index)]
# 		indexint += 1
# 	return result
def donate_to_list(donate_tensor_list,reciever_list,nb_filter_donation,filter_refrence,tensor_donate_sliced_list):
	result = []
	if not type(reciever_list) == list:
		donators =[]
		for i in range(donate_tensor_list.__len__()):
			donate_tensor = donate_tensor_list[i]
			if not donate_tensor.name==reciever_list.name:
				donators+=[tensor_donate_sliced_list[i]]
				# b = donate_tensor[:,:nb_filter_donation,:,:]
				# return merge([b,reciever_list],mode='concat',concat_axis=1)
		donators+=[reciever_list]
		return merge(donators,mode='concat',concat_axis=1,name='leaked_to_'+reciever_list.name)
	else:
		for reciever in reciever_list:
			updated_tensor_list = donate_to_list(donate_tensor_list,reciever,nb_filter_donation,filter_refrence,tensor_donate_sliced_list)
			if type(updated_tensor_list) == list:
				result += [updated_tensor_list]
			else:
				result += [updated_tensor_list]
	return result
def concat_on_list(input_tensor_list,n,layer_index,index = '0'):
	print(input_tensor_list)
	result = []
	indexint = 0
	depth = 1+tree_depth(input_tensor_list)
	if depth == n:
		a = node_list_to_list(input_tensor_list)
		result = merge(a, mode='concat', concat_axis=1, name='Merge_' + str(layer_index) + 'index-' + index + str(
				indexint))
		return result
	for lists_or_tensor in input_tensor_list:
		result+=[concat_on_list(lists_or_tensor,n,layer_index,index + str(indexint))]
		indexint += 1

	return result
def tree_depth(input_tensor_list):
	result=0
	if not type(input_tensor_list)==list:
		return result
	for lists_or_tensor in input_tensor_list:
		if type(lists_or_tensor) == list:
			result = 1+tree_depth(lists_or_tensor)
			break
		else:
			return result

	return result
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