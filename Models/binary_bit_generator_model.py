from keras.layers import Input

from Models.binary_net import gatenet_binary
from Layers.sigmoid_stoch import StochActivation
import keras.backend as K
import numpy as np
def binary_bit_generator(opts, input_shape, nb_classes, input_tensor=None, include_top=True,
                        initialization='glorot_normal'):
	if include_top:
		input_shape = input_shape
	else:
		input_shape = (3, None, None)
	if input_tensor is None:
		img_input = Input(shape=input_shape, name='image_batch')
	else:
		img_input = input_tensor
	x_bit = []
	x_enable = [[]]
	x_bit_out = []
	x_out = []
	x_bit=[]
	for layer_index in np.arange(0,np.ceil(np.log2(nb_classes))):
		x_bit_list = []
		x_enable_list = []
		x_bit_out_list = []
		x_out_node = K.zeros_like([1])
		layer_index = int(layer_index)
		for unit_index in np.arange(0,2**(layer_index)):
			unit_index = int(unit_index)
			x_bit_list+=[gatenet_binary(input_tensor=img_input,opts=opts, input_shape=opts[
				'training_opts']['dataset'][
				'input_shape'],nb_classes=2,output='node')]
			if layer_index==0:
				x_enable_list += [StochActivation(opts,False)(x_bit_list[int(unit_index)][:,0])]
				x_enable_list += [(1 - x_enable_list[2 * unit_index])]
			else:
				x_enable_list+=[StochActivation(opts,False)(x_bit_list[unit_index][:,0])*x_enable[layer_index][
					unit_index]]
				x_enable_list += [(1 - x_enable_list[2 * unit_index]) * x_enable[layer_index][unit_index]]
			x_bit_out_list +=[x_enable_list[2*unit_index][0]]
			# x_bit[layer_index][unit_index] = gatenet_binary(input_tensor=img_input,opts=opts, input_shape=opts[
			# 	'training_opts']['dataset'][
			# 	'input_shape'],nb_classes=2,output='node')
			# x_enable[layer_index+1][2*unit_index] = StochActivation(opts)(x_bit[layer_index][unit_index][
			# 	                                                              0])*x_enable[layer_index][unit_index]
			# x_enable[layer_index+1][2*unit_index+1] = (1-x_enable[layer_index+1][2*unit_index])*x_enable[layer_index][
			# 	unit_index]
			# x_bit_out[layer_index][unit_index] = x_enable[layer_index+1][2*unit_index][0]
			if unit_index==0:
				x_out_node = x_bit_out_list[unit_index]
			else:
				x_out_node+=x_bit_out_list[unit_index]
		x_bit_out+=[x_bit_out_list]
		x_enable+=[x_enable_list]
		x_bit += [x_bit_list]
		x_out+=[x_out_node]
	res =x_out[0]+2*x_out[1]+4*x_out[2]+8*x_out[3]
	res = K.to





	# x_bits1 = gatenet_binary(input_tensor=img_input,opts=opts, input_shape=opts['training_opts']['dataset'][
	# 	'input_shape'],
	#                nb_classes=2,output='node')
	# enable_layer2 = StochActivation(opts)(x_bits1[0])
	# x_bit2_0 = gatenet_binary(input_tensor=img_input,opts=opts, input_shape=opts['training_opts']['dataset'][
	# 	'input_shape'],
	#                nb_classes=2,output='node')
	# x_bit2_1 = gatenet_binary(input_tensor=img_input,opts=opts, input_shape=opts['training_opts']['dataset'][
	# 	'input_shape'],
	#                nb_classes=2,output='node')
	# enable_layer3 = StochActivation(opts)(x_bit2_0[0])
	# x_bit2 = (enable_layer2*x_bit2_0)+((1-enable_layer2)*x_bit2_1)
	#
	# x_bit3_01 =gatenet_binary(input_tensor=img_input,opts=opts, input_shape=opts['training_opts']['dataset'][
	# 	'input_shape'],
	#                nb_classes=2,output='node')
	# x_bit3_02 = gatenet_binary(input_tensor=img_input, opts=opts,
	#                            input_shape=opts['training_opts']['dataset']['input_shape'], nb_classes=2,
	#                            output='node')
	# x_bit3_03 = gatenet_binary(input_tensor=img_input, opts=opts,
	#                            input_shape=opts['training_opts']['dataset']['input_shape'], nb_classes=2,
	#                            output='node')
	# x_bit3_04 = gatenet_binary(input_tensor=img_input, opts=opts,
	#                            input_shape=opts['training_opts']['dataset']['input_shape'], nb_classes=2,
	#                            output='node')
	# x_bit3 =
