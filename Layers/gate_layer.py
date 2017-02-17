import numpy as np
from keras.layers import Convolution2D,Activation
from keras.layers import merge
from Activation.activations import sigmoid_stoch
from Layers.sigmoid_stoch import StochSigmoidActivation,Inverter
from keras.regularizers import l1,l2
from utils.opt_utils import *
import keras.backend as K

from Regularizer.activity_regularizers import VarianceSpatialActivityRegularizer, NormalizeActivityRegularizer


def gate_layer(input_tensor, nb_filter, filter_size, opts, input_shape=(None, None, None), border_mode='valid',
               activity_reg_weight=1,average_reg=.5,stride = (2,2)):
	''' Layer used to gate the convolution output. it will gate the output of a convolutional layer'''
	w_regularizer_str = opts['model_opts']['param_dict']['w_regularizer']['method']
	if w_regularizer_str == 'l1':
		w_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
	if w_regularizer_str == None:
		w_reg = None
	if w_regularizer_str=='l2':
		w_reg= l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
	dict_regularizer = {
		'variance_spatial': VarianceSpatialActivityRegularizer,
		'mean_value'      : NormalizeActivityRegularizer}

	activity_regularizer = dict_regularizer.get(opts['model_opts']['param_dict']['gate_layer'][
		                                            'gate_activity_regularizer']['method'])
	# stochActive = StochSigmoidActivation()
	reg_opt_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict']
	gate_activation = opts['model_opts']['param_dict']['gate_layer']['gate_activation']['method']
	data_activation = opts['model_opts']['param_dict']['data_layer']['data_activation']['method']
	if opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'] == 'variance_spatial':
		reg_opt_dict['shape'] = (nb_filter, input_shape[1], input_shape[2])
	# if activity_regularizer== None:
	# 	gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation=gate_activation,
	# 	                            input_shape=input_shape,
	# 	                            border_mode=border_mode,W_regularizer=w_reg)(input_tensor)
	# else:
	# 	gate_output = Convolution2D(nb_filter, filter_size, filter_size, activation=gate_activation,
	# 	                            input_shape=input_shape,
	# 	                            border_mode=border_mode,
	# 	                            activity_regularizer=activity_regularizer(reg_opt_dict,activity_reg_weight,
	# 	                                                                      average_reg),
	# 	                            W_regularizer=w_reg)(input_tensor)
	# gate_output= StochSigmoidActivation()(gate_output)
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
	                          border_mode=border_mode,W_regularizer=w_reg)(input_tensor)
	gate_output = Activation(activation=gate_activation)(data_conv)
	if get_stoc(opts):
		# Stoch tanh
		# gate_output = Activation(activation=gate_activation)(data_conv)
		# gate_output = StochSigmoidActivation(opts,tan=True)(gate_output)
		# data_conv = merge([data_conv, gate_output], mode='mul')
		gate_output = Activation(activation=gate_activation)(data_conv)
		gate_output = StochSigmoidActivation(opts,tan=False)(gate_output)
		gate_output_inv = Inverter()(gate_output)
		inv_pas = merge([gate_output_inv, data_conv], mode='mul')
		pas = merge([gate_output, data_conv], mode='mul')
		# Concatt Experiment
		# one = K.ones_like(gate_output)
		# gate_inverted = one-gate_output
		# gate_output = merge([gate_output, gate_inverted], mode='concat', concat_axis=1)
		# data_conv= merge([data_conv,data_conv],mode='concat',concat_axis=1)
		# gate_output = K.concatenate([gate_output,gate_output],axis=1)
		# data_conv = K.concatenate([data_conv, data_conv], axis=1)
	merged = merge([inv_pas, pas], mode='concat',concat_axis=1)
	return merged


def gated_layers_sequence(input_tensor, total_layers, nb_filter, filter_size, input_shape=(None, None, None)):
	out = input_tensor
	for i in np.arange(0, total_layers):
		out = gate_layer(out, nb_filter, filter_size, input_shape)
	return out
