import keras
from keras import optimizers

from modeldatabase.Binary_models import model_constructor_utils
from modeldatabase.Binary_models.model_db import get_model_from_db
from utils.gen_utils import *
from utils.modelutils import model_modification_utils
from utils.opt_utils import *
from utils.trainingutils.training_phases_utils import *


def check_model_list(model_list, datasets):
	opts = default_opt_creator()
	for dataset_str in datasets:
		for model in model_list:
			set_dataset(opts, dataset=dataset_str)
			set_default_opts_based_on_model_dataset(opts)
			set_model_string(opts, model)
			get_model_from_db(model, opts)


if __name__ == '__main__':
	# gatenet_binary_merged_model lenet_amir ,gatenet_binary_model
	vanila_models = ['nin_relu_baseline_caffe']
	dataset_str = 'cifar10'
	#
	weight_model_experiment_name = get_experiment_name_prompt('please select the experiment for the model you want to load weight from[to load '
															  'weight from]')
	experiment_name = 'load_weight'
	print((keras.__version__))
	model_str = vanila_models[0]
	cont = True
	while (cont):
		weight_model_name = get_model_from_experiment_prompt(weight_model_experiment_name, dataset_str)
		print(100 * '*', 3 * '\n', model_str, '\n', dataset_str, 3 * '\n', 100 * '*')
		opts = default_opt_creator()
		opts['experiment_name'] = experiment_name
		opts['experiment_tag'] = experiment_name + '/' + dataset_str + '/' + weight_model_name + 'loaded_to_' + model_str
		set_dataset(opts, dataset=dataset_str)
		opts = set_model_string(opts, model_str)
		opts = set_default_opts_based_on_model_dataset(opts)
		input_shape = opts['training_opts']['dataset']['input_shape']
		nb_class = opts['training_opts']['dataset']['nb_classes']
		# opts = set_expand_rate(opts, param_expand_sel)
		# optimizer = optimizers.Nadam()
		optimizer = optimizers.SGD(lr=opts['optimizer_opts']['lr'], momentum=opts['optimizer_opts']['momentum'],
								   decay=opts['optimizer_opts']['decay'], nesterov=opts['optimizer_opts']['nestrov'])
		# optimizer = optimizers.Adadelta()
		""" MODEL PREPARE """
		model = get_model_from_db(model_str, opts)
		weight_model = get_model_from_db(weight_model_name, opts)
		model_modification_utils.load_weights_by_block_index_list(model, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], os.path.join(
			global_constant_var.get_experimentcase_abs_path(weight_model_experiment_name, dataset_str, weight_model_name), 'checkpoint'),
																  model_constructor_utils.CONVSH_NAME)
		model_modification_utils.load_weights_by_block_index_list(weight_model, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], os.path.join(
			global_constant_var.get_experimentcase_abs_path(weight_model_experiment_name, dataset_str, weight_model_name), 'checkpoint'),
																  model_constructor_utils.CONVSH_NAME)
		model.compile(loss=opt_utils.get_loss(opts), optimizer=optimizer, metrics=opt_utils.get_metrics(opts))
		weight_model.compile(loss=opt_utils.get_loss(opts), optimizer=optimizer, metrics=opt_utils.get_metrics(opts))
		method_names = find_key_value_to_str_recursive(opts, '', {'param_expand'})
		opts['experiment_name'] = method_names
		# LOAD DATA
		(data_train, label_train), (data_test, label_test) = load_data(dataset_str, opts)
		data_train, data_test = preprocess_data_phase(opts, data_train, data_test)
		data_gen = data_augmentation_phase(opts)
		# COLLECT CALLBACKS
		callback_list = []
		# result_manager = PlotMetrics(opts)
		# experiments_abs_path = result_manager.history_holder.dir_abs_path
		# callback_list += [result_manager]
		# callback_list += [TensorboardVisualizer(log_dir=experiments_abs_path + '/logs', histogram_freq=1, write_graph=True,
		# write_images=False)]
		callback_list = collect_callbacks(opts)
		# TRAIN
		samples_per_epoch = data_train.shape[0] if opts['training_opts']['samples_per_epoch'] == -1 else opts['training_opts']['samples_per_epoch']
		results_target_model = model.evaluate(data_test, label_test)
		results_weight_model = weight_model.evaluate(data_test, label_test)
		print('\n')
		print(*model.metrics_names, sep='\t\t')
		print('\n Results from vanila Model {}\n'.format(model_str))
		print(*results_target_model, sep='\t\t')
		print('\n results from weight model {}\n'.format(weight_model_name))
		print(*results_weight_model, sep='\t\t')
		a = input('do you want to evaluate another model?[y/n/b for back]')
		if a == '' or a == 'y':
			cont = True
		elif a=='b':
			weight_model_experiment_name = get_experiment_name_prompt(
				'please select the experiment for the model you want to load weight from[to load '
				'weight from]')
			cont = True
		else:
			cont = False
		# model.evaluate(
		# 	data_gen.flow(data_train, label_train, batch_size=opts['training_opts']['batch_size'], shuffle=True, seed=opts['seed']),
		# 	samples_per_epoch=0, nb_epoch=2, callbacks=callback_list, validation_data=(data_test, label_test))
