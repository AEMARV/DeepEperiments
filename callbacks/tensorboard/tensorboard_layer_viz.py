import warnings

import keras.backend as K
import tensorflow as tf
from keras.backend.common import image_dim_ordering

from callbacks.tensorboard import tensor_board_utills
from callbacks.tensorboard.tensor_board_utills import layer_output_to_imgtensor

def multichannel_tensor_image_visualizer(tensor,filter_num_to_show_max,collection=None):
	tensor_list = [tensor]
	image_tile_tensor = []
	for index, tensor in enumerate(tensor_list):
		image_tile_tensor += [layer_output_to_imgtensor(tensor, filter_num_to_show_max)]
		if index == 2:
			break
	return K.concatenate(3*image_tile_tensor,axis=-1)
def activation_map_image_visualizer(layer, filter_num_to_show_max,collection=None):
	tensor_list = layer.output if type(layer.output_shape) is list else tensor_board_utills.get_outbound_tensors_as_list(layer)
	image_tile_tensor = []
	if tensor_list.__len__() < 3:
		for index, tensor in enumerate(tensor_list):
			image_tile_tensor += [layer_output_to_imgtensor(tensor, filter_num_to_show_max)]
			if index == 2:
				break
		pad_channel = tf.zeros_like(image_tile_tensor[0])
		image_tile_tensor = image_tile_tensor + (2 - index) * [pad_channel]
		image = K.concatenate(image_tile_tensor, axis=-1)
		return [tf.summary.image('{}_out'.format(layer.name), image,collections=collection)]
	if tensor_list.__len__() >= 3:
		for index, tensor in enumerate(tensor_list):
			image_tile_tensor += [layer_output_to_imgtensor(tensor, filter_num_to_show_max)]
		pad_channel = tf.zeros_like(image_tile_tensor[0])
		res =[]
		index =0
		while index<len(image_tile_tensor):
			end = min(index+3,len(image_tile_tensor))
			image_tile_tensor_block = image_tile_tensor[index:end]+(index+3-end)*[pad_channel]
			image = K.concatenate(image_tile_tensor_block, axis=-1)
			res+= [tf.summary.image('{}_{}:{}out'.format(layer.name,index,end), image, collections=collection)]
			index+=3
		return res



def ReLU_histogram(layer,collection=None):
	tensor_list = tensor_board_utills.get_outbound_tensors_as_list(layer)
	for tensor in tensor_list:
		tf.summary.histogram(name='{}_BER_Branch_Merged_Histogram'.format(layer.name), values=tensor,collections=collection)


def BeReLU_histogram(layer,collection=None):
	tensor_list = tensor_board_utills.get_outbound_tensors_as_list(layer)
	for index, tensor in enumerate(tensor_list):
		add_tensor = tensor if add_tensor is None else add_tensor - tensor
		trinary_tensor = (tensor / (tensor + K.epsilon())) if add_tensor is None else add_tensor - (tensor / (tensor + K.epsilon()))
	tf.summary.histogram(name='{}_BER_Branch_Merged_Histogram'.format(layer.name), values=add_tensor,collections=collection)
	tf.summary.histogram(name='{}_BER_Median_Merged_Histogram_Trinary'.format(layer.name), value=trinary_tensor,collections=collection)
