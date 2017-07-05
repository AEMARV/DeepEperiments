import os

import h5py
import keras.backend as K
from keras.engine.topology import preprocess_weights_for_loading

from modeldatabase.Binary_models import model_constructor_utils
from utils.global_constant_var import get_experimentcase_abs_path


def get_experiment_checkpoint_path(experiment_name, dataset, model,model_parameters=None,run_number=1):
	return os.path.join(get_experimentcase_abs_path(experiment_name, dataset, model,model_desc=model_parameters,run_num=run_number), 'checkpoint')


def freeze_layer(model, layer_name_list):
	for layer in model.layers:
		if layer.name in layer_name_list:
			layer.trainable = False


def freezeunfreeze_layer_by_block_num_list(model, blocks, layer_name_rule=model_constructor_utils.CONVSH_NAME, freeze=True):
	layer_names_exluding_index = [layer_name_rule.format(block, '') for block in blocks]
	for layer in model.layers:
		for layer_target in layer_names_exluding_index:
			if not layer.name.find(layer_target) == -1:
				layer.trainable = not freeze


def load_weights_by_layers(layers, filepath):
	if h5py is None:
		raise ImportError('`load_weights` requires h5py.')
	f = h5py.File(filepath, mode='r')
	if 'layer_names' not in f.attrs and 'model_weights' in f:
		f = f['model_weights']

	load_weights_from_hdf5_group_by_name(f, layers)
	if hasattr(f, 'close'):
		f.close()


def load_weights_by_block_index_list(model, blocks, filepath, layer_name_rule=model_constructor_utils.CONVSH_NAME):
	layer_names_exluding_index = [layer_name_rule.format(block, '') for block in blocks]
	layers = []
	for layer in model.layers:
		for layer_target in layer_names_exluding_index:
			if not layer.name.find(layer_target) == -1:
				layers += [layer]
	load_weights_by_layers(layers, filepath)


def load_weights_from_hdf5_group_by_name(f, layers):
	if 'keras_version' in f.attrs:
		original_keras_version = f.attrs['keras_version'].decode('utf8')
	else:
		original_keras_version = '1'
	if 'backend' in f.attrs:
		original_backend = f.attrs['backend'].decode('utf8')
	else:
		original_backend = None

	# New file format.
	layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

	# Reverse index of layer name to list of layers with name.
	index = {}
	for layer in layers:
		if layer.name:
			index.setdefault(layer.name.split('-')[0], []).append(layer)
	# We batch weight value assignments in a single backend call
	# which provides a speedup in TensorFlow.
	weight_value_tuples = []
	for k, name in enumerate(layer_names):
		g = f[name]
		weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
		weight_values = [g[weight_name] for weight_name in weight_names]
		for layer in index.get(name, []):
			symbolic_weights = layer.weights
			weight_values = preprocess_weights_for_loading(layer, weight_values, original_keras_version, original_backend)
			if len(weight_values) != len(symbolic_weights):
				raise ValueError('Layer #' + str(k) + ' (named "' + layer.name + '") expects ' + str(len(symbolic_weights)) + ' weight(s), but the saved weights' + ' have ' + str(
					len(weight_values)) + ' element(s).')
			# Set values.
			for i in range(len(weight_values)):
				weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
	K.batch_set_value(weight_value_tuples)
