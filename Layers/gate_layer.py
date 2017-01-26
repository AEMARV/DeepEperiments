import numpy as np
from keras.layers import Convolution2D,Activation
from keras.layers import merge
from Activation.activations import sigmoid_stoch
from Layers.sigmoid_stoch import StochSigmoidActivation

from Regularizer.activity_regularizers import VarianceSpatialActivityRegularizer, NormalizeActivityRegularizer


def gate_layer(input_tensor, nb_filter, filter_size, opts, input_shape=(None, None, None), border_mode='valid'):
	''' Layer used to gate the convolution output. it will gate the output of a convolutional layer'''
	dict_regularizer = {
		'variance_spatial': VarianceSpatialActivityRegularizer,
		'mean_value'      : NormalizeActivityRegularizer}

	activity_regularizer = dict_regularizer.get(opts['model_opts']['param_dict']['gate_layer'][
		                                            'gate_activity_regularizer']['method'])

	reg_opt_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict']
	gate_activation = opts['model_opts']['param_dict']['gate_layer']['gate_activation']['method']
	data_activation = opts['model_opts']['param_dict']['data_layer']['data_activation']['method']
	if opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'] == 'variance_spatial':
		reg_opt_dict['shape'] = (nb_filter, input_shape[1], input_shape[2])
	if activity_regularizer== None:
		gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation=gate_activation,
		                            input_shape=input_shape,
		                            border_mode=border_mode)(input_tensor)
	else:
		gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation=gate_activation,
		                            input_shape=input_shape,
		                            border_mode=border_mode,
		                            activity_regularizer=activity_regularizer(reg_opt_dict))(input_tensor)
	# gate_output = StochSigmoidActivation()(gate_output)
	# if opts['model_opts']['regularizer'] == 'variance_spatial':
	# 	gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation='sigmoid',
	#
	# 	                            input_shape=input_shape, border_mode=border_mode,
	# 	                            activity_regularizer=activity_regularizers.VarianceSpatialActivityRegularizer(
	# 		                            opts['model_opts']['alpha_activity'],
	# 		                            (nb_filter, input_shape[1], input_shape[2])))(input_tensor)
	# # No activity Regularizer
	# else:
	# 	gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation='softplus',
	# 	                            input_shape=input_shape,
	# 	                            border_mode=border_mode, )(input_tensor)
	data_conv = Convolution2D(nb_filter, filter_size, filter_size, activation=data_activation, input_shape=input_shape,
	                          border_mode=border_mode)(input_tensor)
	merged = merge([data_conv, gate_output], mode='mul')
	return merged


def gated_layers_sequence(input_tensor, total_layers, nb_filter, filter_size, input_shape=(None, None, None)):
	out = input_tensor
	for i in np.arange(0, total_layers):
		out = gate_layer(out, nb_filter, filter_size, input_shape)
	return out
