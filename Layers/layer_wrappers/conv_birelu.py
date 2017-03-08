from keras.layers import Convolution2D,Activation,merge
from Layers.binary_layers.fundamental_layers_binary import Inverter,Negater,StochActivation
from utils.opt_utils import *
from Layers.binary_layers.birelu import Birelu,Relu,Birelu_nary
def conv_birelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor,relu_birelu_switch=1):
	"this function is passing data through a convolution and then pass it through the birelu (doubles channels)"
	'returns list'
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None,
	                          input_shape=input_shape, border_mode=border_mode, W_regularizer=w_reg,
	                          name='E_conv_exp_'+'nbfilter-'+str(nb_filter)+'_layer'+str(layer_index)+'_index-'+str(
		                          index))(
		input_tensor)
	output_tensor_list = Birelu(gate_activation,relu_birelu_sel=relu_birelu_switch,name='E_Birelu_layer-'+str(
		layer_index)+'_index-'+str(index),layer_index=layer_index)(
		data_conv)
	return output_tensor_list
def conv_birelunary_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor,relu_birelu_switch=1):
	"this function is passing data through a convolution and then pass it through the birelu (doubles channels)"
	'returns list'
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None,
	                          input_shape=input_shape, border_mode=border_mode, W_regularizer=w_reg,
	                          name='E_conv_exp_'+'nbfilter-'+str(nb_filter)+'_layer'+str(layer_index)+'_index-'+str(
		                          index))(
		input_tensor)
	output_tensor_list = Birelu_nary(gate_activation,relu_birelu_sel=relu_birelu_switch,name='E_Birelunary_layer-'+str(
		layer_index)+'_index-'+str(index),layer_index=layer_index,nb_filter=nb_filter)(
		data_conv)
	return output_tensor_list
def conv_birelu_swap(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor_list,relu_birelu_switch=1):
	tensor_concat =merge([input_tensor_list[0], input_tensor_list[1]], mode='concat', concat_axis=1,
	                     name='S_MergeInput_layer'+str(
		layer_index)+'_index-'+str(
		index))
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None,
	                          input_shape=input_shape, border_mode=border_mode, W_regularizer=w_reg,
	                          name='S_conv_Swap_'+'nbfilter-'+str(nb_filter)+'_layer-'+str(layer_index)+'number'+str(
		                          index))(
		tensor_concat)
	output_tensor_list = Birelu(gate_activation,relu_birelu_sel=relu_birelu_switch,name='S_BiRelu_layer-'+str(
		layer_index)+'_index'+str(index))(
		data_conv)
	return output_tensor_list
def conv_relu_merge(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor_list):
	tensor_concat = merge([input_tensor_list[0], input_tensor_list[1]], mode='concat', concat_axis=1,
	                      name='RM_concat_Merge_layer=' + str(layer_index) +'index-'+ str(index))


	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None, input_shape=input_shape,
	                          border_mode=border_mode, W_regularizer=w_reg,
	                          name='RM_conv_'+'nbfilter'+str(nb_filter) +'_layer-'+str(layer_index) + '_index' +
	                               str(
		index))(tensor_concat)
	output_tensor = Relu(gate_activation,name='RM_RELU_layer-'+str(layer_index)+'_index'+str(index))(data_conv)
	return output_tensor
def conv_birelu_merge(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor_list):
	tensor_concat = merge([input_tensor_list[0], input_tensor_list[1]], mode='concat', concat_axis=1,
	                      name='BM_Merge_input_layer=' + str(layer_index) +'index-'+ str(index))


	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None, input_shape=input_shape,
	                          border_mode=border_mode, W_regularizer=w_reg,
	                          name='BM_conv_Merge_'+'nbfilter'+str(nb_filter) +'_layer-'+str(layer_index) + '_index' +
	                               str(
		index))(tensor_concat)
	output_tensor = Birelu(gate_activation,name='BM_Birelu_layer-'+str(layer_index)+'_index'+str(index))(data_conv)
	output_tensor = merge([output_tensor[0], output_tensor[1]], mode='concat', concat_axis=1,
	                      name='BM_Merge_output_layer=' + str(layer_index) + 'index-' + str(index))
	return output_tensor
def conv_relu(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor):
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None, input_shape=input_shape,
	                          border_mode=border_mode, W_regularizer=w_reg,
	                          name='Conv_'+'nbfilter'+str(nb_filter) +'_layer-'+str(layer_index) + '_index' +
	                               str(
		index))(input_tensor)
	output_tensor = Relu(gate_activation,name='R_relu_layer-'+str(layer_index)+'_index'+str(index))(data_conv)
	return output_tensor
def concat(input_tensor_list,index,layer_index):
	tensor_concat = merge([input_tensor_list[0], input_tensor_list[1]], mode='concat', concat_axis=1,
	                      name='Merge_' + str(layer_index) + 'index-' + str(index))
	return tensor_concat