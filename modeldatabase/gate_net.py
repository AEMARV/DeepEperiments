import numpy as np
from Layers.gate_layer import gate_layer
from keras import backend as K
from keras.engine import Model
from keras.layers import Input, Flatten, Dense, Dropout, Activation, AveragePooling2D, MaxPooling2D, Convolution2D
from keras.optimizers import SGD
from keras.regularizers import l1, l2
from keras.utils.visualize_util import plot

from utils.modelutils.layers.dropout_gated import droupout_gated_layer
from utils.opt_utils import get_filter_size


def vgg_gated(include_top=True, weights=None, input_tensor=None, opts=None, initialization='glorot_normal'):
	'''Instantiate the VGG16 architecture,
	optionally loading weights pre-trained
	on ImageNet. Note that when using TensorFlow,
	for best performance you should set
	`image_dim_ordering="tf"` in your Keras config
	at ~/.keras/keras.json.

	The model and the weights are compatible with both
	TensorFlow and Theano. The dimension ordering
	convention used by the model is the one
	specified in your Keras config file.

	# Arguments
		include_top: whether to include the 3 fully-connected
			layers at the top of the network.
		weights: one of `None` (random initialization)
			or "imagenet" (pre-training on ImageNet).
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as image input for the model.

	# Returns
		A Keras model instance.
	'''
	# Determine proper input shape
	if K.image_dim_ordering() == 'th':
		if include_top:
			input_shape = (3, 224, 224)
		else:
			input_shape = (3, None, None)
	else:
		if include_top:
			input_shape = (224, 224, 3)
		else:
			input_shape = (None, None, 3)

	img_input = Input(shape=input_shape, name='image_batch')
	# Block 1
	x = gate_layer(img_input, 64, 3, input_shape=input_shape, opts=opts, border_mode='same')
	# x = gate_layer(x, 64, 3, input_shape=(64,input_shape[1],input_shape[2]), opts=opts, border_mode='same')
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = gate_layer(x, 128, 3, input_shape=(64, (input_shape[1] / 2), (input_shape[2]) / 2), opts=opts,
	               border_mode='same')
	# x = gate_layer(x, 128, 3, input_shape=(128,(input_shape[1]/2),(input_shape[2])/2), opts=opts,
	#                border_mode='same')
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
	#
	# # Block 3
	x = gate_layer(x, 256, 3, input_shape=(128, (input_shape[1] / 4), (input_shape[2]) / 4), opts=opts,
	               border_mode='same')
	# x = gate_layer(x, 256, 3, input_shape=(256,(input_shape[1]/4),(input_shape[2])/4), opts=opts,
	#                border_mode='same')
	# x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3',init=initialization)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
	#
	# # Block 4
	x = gate_layer(x, 512, 3, input_shape=(256, (input_shape[1] / 8), (input_shape[2]) / 8), opts=opts,
	               border_mode='same')
	# x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2',init=initialization)(x)
	# x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3',init=initialization)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
	#
	# Block 512
	x = gate_layer(x, 512, 3, input_shape=(512, (input_shape[1] / 16), (input_shape[2]) / 16), opts=opts,
	               border_mode='same')
	# x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2',init=initialization)(x)
	# x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3',init=initialization)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	if include_top:
		# Classification block
		x = Flatten(name='flatten')(x)
		x = Dense(4096, activation='relu', name='fc1', init=initialization)(x)
		x = Dropout(p=.5)(x)
		x = Dense(4096, activation='relu', name='fc2', init=initialization)(x)
		x = Dropout(p=.5)(x)
		x = Dense(20, activation='softmax', name='predictions', init=initialization)(x)

	# Create model
	model = Model(img_input, x)
	model.summary()
	return model




def gatenet_amir(opts, input_shape, nb_classes, input_tensor=None, include_top=True, initialization='glorot_normal'):
	if include_top:
		input_shape = input_shape
	else:
		input_shape = (3, None, None)
	if input_tensor is None:
		img_input = Input(shape=input_shape, name='image_batch')
	else:
		img_input = input_tensor
	w_regularizer_str = opts['model_opts']['param_dict']['w_regularizer']['method']
	if w_regularizer_str == 'l1':
		w_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
	if w_regularizer_str is None:
		w_reg = None
	if w_regularizer_str == 'l2':
		w_reg = l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	average_reg_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'][
		'average_reg']
	average_reg_weight_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'][
		'param_dict'][
		'average_reg_weight']
	data_activation = opts['model_opts']['param_dict']['data_layer']['data_activation']['method']
	filter_size = get_filter_size(opts)
	if filter_size == -1:
		f_size = [5, 3, 5, 4]
	else:
		f_size = np.min([[32, 16, 7, 3], [filter_size, (filter_size + 1) / 2, filter_size, filter_size - 1]], 0)
	# Layer 1 Conv 5x5 32ch  border 'same' Max Pooling 3x3 stride 2 border valid
	x = gate_layer(img_input, int(32 * expand_rate), f_size[0], input_shape=input_shape, opts=opts, border_mode='same',
	               activity_reg_weight=average_reg_weight_dict[0], average_reg=average_reg_dict[0])
	### ADDED
	# x = gate_layer(x, int(32 * expand_rate), f_size[0], input_shape=input_shape, opts=opts, border_mode='same',
	#                activity_reg_weight=average_reg_weight_dict[0], average_reg=average_reg_dict[0])
	# x = gate_layer(x, int(32 * expand_rate), f_size[0], input_shape=input_shape, opts=opts, border_mode='same',
	#                activity_reg_weight=average_reg_weight_dict[0], average_reg=average_reg_dict[0])
	# TODO add stride for conv layer
	x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(x)

	#                           Layer 2 Conv 5x5 32ch  border 'same' AveragePooling 3x3 stride 2
	x = gate_layer(x, int(32 * expand_rate), f_size[1], input_shape=(32, (input_shape[1] - 2), (input_shape[2]) - 2),
	               opts=opts, activity_reg_weight=average_reg_weight_dict[1], average_reg=average_reg_dict[1],
	               border_mode='same')
	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(x)

	#                           Layer 3 Conv 5x5 128ch  border 'same' AveragePooling3x3 stride 2
	x = gate_layer(x, int(128 * expand_rate), f_size[2],
	               input_shape=(32, (input_shape[1] - 2) / 2, (input_shape[2] - 2) / 2), opts=opts, border_mode='same',
	               activity_reg_weight=average_reg_weight_dict[2], average_reg=average_reg_dict[2])
	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(x)

	#                           Layer 4 Conv 4x4 64ch  border 'same' no pooling
	x = gate_layer(x, int(64 * expand_rate), f_size[3],
	               input_shape=(64, ((input_shape[1] - 2) / 2) - 2, ((input_shape[2] - 2) / 2) - 2), opts=opts,
	               border_mode='valid', activity_reg_weight=average_reg_weight_dict[3],
	               average_reg=average_reg_dict[3])
	if not include_top:
		model = Model(input=img_input, output=x)
	else:
		x = Flatten(name='flatten')(x)
		x = Dense(nb_classes)(x)
		x = Activation('softmax')(x)
		model = Model(input=img_input, output=x)
	return model


def dropout_gated_lenet(opts, input_shape, nb_classes, input_tensor=None, include_top=True,
                        initialization='glorot_normal'):
	if include_top:
		input_shape = input_shape
	else:
		input_shape = (3, None, None)
	if input_tensor == None:
		img_input = Input(shape=input_shape, name='image_batch')
	else:
		img_input = input_tensor
	w_regularizer_str = opts['model_opts']['param_dict']['w_regularizer']['method']
	if w_regularizer_str == 'l1':
		w_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
	if w_regularizer_str == None:
		w_reg = None
	if w_regularizer_str == 'l2':
		w_reg = l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	average_reg_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'][
		'average_reg']
	average_reg_weight_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'][
		'param_dict'][
		'average_reg_weight']
	data_activation = opts['model_opts']['param_dict']['data_layer']['data_activation']['method']
	x = Convolution2D(32 * expand_rate, 3, 3, activation=data_activation, input_shape=input_shape, border_mode='same',
	                  W_regularizer=w_reg)(img_input)
	raw_conv = Convolution2D(32 * expand_rate, 3, 3, activation=data_activation, border_mode='valid',
	                         W_regularizer=w_reg)(x)
	data_ready = Activation(activation=data_activation)(raw_conv)
	data_ready = MaxPooling2D(pool_size=(2, 2))(data_ready)
	raw_conv = MaxPooling2D(pool_size=(2, 2))(raw_conv)
	# x = Dropout(0.25)(x)
	x = droupout_gated_layer(input_tensor=x, filter_size=3, opts=opts, input_shape=x._shape_as_list(),
	                         activity_reg_weight=1, average_reg=.75)

	x = Convolution2D(64 * expand_rate, 3, 3, activation=data_activation, border_mode='same', W_regularizer=w_reg)(x)
	y = Convolution2D(64 * expand_rate, 3, 3, activation=None, border_mode='valid', W_regularizer=w_reg)(x)
	x = Activation(activation=data_activation)(y)
	# x = activations('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	y = MaxPooling2D(pool_size=(2, 2))(y)
	# x = Dropout(0.25)(x)
	x = droupout_gated_layer(input_tensor=y, filter_size=3, opts=opts, input_shape=x._shape_as_list(),
	                         activity_reg_weight=.5, average_reg=.75)
	if not include_top:
		model = Model(input=img_input, output=x)
	else:
		x = Flatten(name='flatten')(x)
		x = Dense(512)(x)
		x = Activation('relu')(x)
		x = Dropout(.5)(x)
		x = Dense(nb_classes)(x)
		x = Activation('softmax')(x)
		model = Model(input=img_input, output=x)
	return model


def gated_lenet(opts, input_shape, nb_classes, input_tensor=None, include_top=True, initialization='glorot_normal'):
	if include_top:
		input_shape = input_shape
	else:
		input_shape = (3, None, None)
	if input_tensor == None:
		img_input = Input(shape=input_shape, name='image_batch')
	else:
		img_input = input_tensor
	w_regularizer_str = opts['model_opts']['param_dict']['w_regularizer']['method']
	if w_regularizer_str == 'l1':
		w_reg = l1(opts['model_opts']['param_dict']['w_regularizer']['value'])
	if w_regularizer_str == None:
		w_reg = None
	if w_regularizer_str == 'l2':
		w_reg = l2(opts['model_opts']['param_dict']['w_regularizer']['value'])
	expand_rate = opts['model_opts']['param_dict']['param_expand']['rate']
	average_reg_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'][
		'average_reg']
	average_reg_weight_dict = opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'][
		'param_dict'][
		'average_reg_weight']
	data_activation = opts['model_opts']['param_dict']['data_layer']['data_activation']['method']
	# x = gate_layer(img_input, int(32*expand_rate/2), 3, input_shape=input_shape,opts=opts,border_mode='same',
	#                activity_reg_weight=average_reg_weight_dict[0],average_reg=average_reg_dict[0])
	#
	# x = gate_layer(x, int(32*expand_rate/2), 3,input_shape=(32,(input_shape[1]-2),(input_shape[2])-2),opts=opts,
	#                activity_reg_weight=average_reg_weight_dict[1],average_reg=average_reg_dict[1])
	x = Convolution2D(32 * expand_rate, 3, 3, activation=data_activation, input_shape=input_shape, border_mode='same',
	                  W_regularizer=w_reg)(img_input)
	x = Convolution2D(32 * expand_rate, 3, 3, activation=data_activation, border_mode='valid', W_regularizer=w_reg)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# x = Dropout(0.25)(x)

	x = gate_layer(x, int(64 * expand_rate / 2), 3,
	               input_shape=(32, (input_shape[1] - 2) / 2, (input_shape[2] - 2) / 2), opts=opts, border_mode='same',
	               activity_reg_weight=average_reg_weight_dict[2], average_reg=average_reg_dict[2])

	x = gate_layer(x, int(64 * expand_rate / 2), 3,
	               input_shape=(64, ((input_shape[1] - 2) / 2) - 2, ((input_shape[2] - 2) / 2) - 2), opts=opts,
	               border_mode='valid', activity_reg_weight=average_reg_weight_dict[3],
	               average_reg=average_reg_dict[3])
	# x = activations('relu')(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	# x = Dropout(0.25)(x)
	if not include_top:
		model = Model(input=img_input, output=x)
	else:
		x = Flatten(name='flatten')(x)
		x = Dense(512)(x)
		x = Activation('relu')(x)
		x = Dropout(.5)(x)
		x = Dense(nb_classes)(x)
		x = Activation('softmax')(x)
		model = Model(input=img_input, output=x)
	return model


if __name__ == '__main__':
	model = gated_lenet(include_top=True)
	model.summary()
	plot(model, to_file='model.png')
	sgd = SGD(lr=0.01, momentum=.9, decay=5 * 1e-4, nesterov=True)
	model.compile(optimizer=sgd, loss='mse')
	np.random.seed(0)
	input_im = np.random.randint(0, 256, (1, 3, 4, 4))
	print "input data: \n", input_im
	a = model.predict(input_im)
	print "Predictions"
	print a
	print a.shape
