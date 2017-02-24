from keras.layers import Convolution2D,Activation,merge
from Layers.binary_layers.fundamental_layers_binary import Inverter,Negater,StochActivation
from utils.opt_utils import *
from Layers.binary_layers.birelu import Birelu,Relu
def conv_birelu_expand(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor):
	"this function is passing data through a convolution and then pass it through the birelu (doubles channels)"
	'returns list'
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None,
	                          input_shape=input_shape, border_mode=border_mode, W_regularizer=w_reg,
	                          name='conv_exp_'+'nbfilter-'+str(nb_filter)+'_layer'+str(layer_index)+'_index-'+str(
		                          index))(
		input_tensor)
	output_tensor_list = Birelu(gate_activation,name='BireluExp_layer-'+str(layer_index)+'_index-'+str(index))(
		data_conv)
	return output_tensor_list
def conv_birelu_swap(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor_list):
	tensor_concat =merge([input_tensor_list[0], input_tensor_list[1]], mode='concat', concat_axis=1,
	                     name='concatSwap_layer'+str(
		layer_index)+'_index-'+str(
		index))
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None,
	                          input_shape=input_shape, border_mode=border_mode, W_regularizer=w_reg,
	                          name='conv_Swap_'+'nbfilter-'+str(nb_filter)+'_layer-'+str(layer_index)+'number'+str(
		                          index))(
		tensor_concat)
	output_tensor_list = Birelu(gate_activation,name='BireluSwap_layer-'+str(layer_index)+'_index'+str(index))(
		data_conv)
	return output_tensor_list
def conv_relu_merge(nb_filter,filter_size,border_mode,input_shape,w_reg,gate_activation,index,layer_index,
                      input_tensor_list):
	tensor_concat = merge([input_tensor_list[0], input_tensor_list[1]], mode='concat', concat_axis=1,
	                      name='concat_Merge_layer=' + str(layer_index) +'index-'+ str(index))


	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=None, input_shape=input_shape,
	                          border_mode=border_mode, W_regularizer=w_reg,
	                          name='conv_Merge_'+'nbfilter'+str(nb_filter) +'_layer-'+str(layer_index) + '_index' +
	                               str(
		index))(tensor_concat)
	output_tensor = Relu(gate_activation,name='ReluMerge_layer-'+str(layer_index)+'_index'+str(index))(data_conv)
	return output_tensor