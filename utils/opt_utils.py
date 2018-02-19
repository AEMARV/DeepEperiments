import modeldatabase
from utils.modelutils.layers.kldivg.layers import *


def set_loss(opts, loss_string):
	opts['optimizer_opts']['loss']['method'] = loss_string
	return opts


def set_weight_decay(opts, decay):
	opts['optimizer_opts']['decay'] = decay
	return opts


def get_nb_classes(opts):
	return opts['training_opts']['dataset']['nb_classes']


def get_input_shape(opts):
	return opts['training_opts']['dataset']['input_shape']


def set_lr(opts, lr):
	opts['optimizer_opts']['lr'] = lr
	return opts


def set_model_string(opts, method_str):
	opts['model_opts']['method'] = method_str
	return opts


def set_stoch(opts, stoch_enable):
	opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch']['method'] = stoch_enable
	return opts


def get_metrics(opts):
	return opts['training_opts']['metrics']


def get_loss(opts):
	return opts['optimizer_opts']['loss']['method']


def get_model_string(opts):
	return opts['model_opts']['method']


def set_expand_rate(opts, expand_rate):
	opts['model_opts']['param_dict']['param_expand']['rate'] = expand_rate
	return opts


def get_expand_rate(opts):
	return opts['model_opts']['param_dict']['param_expand']['rate']


def set_dataset(opts, dataset):
	opts['training_opts']['dataset']['method'] = dataset
	if opts['training_opts']['dataset']['method'] == 'cifar100':
		opts['training_opts']['dataset']['nb_classes'] = 100
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
		return opts
	if opts['training_opts']['dataset']['method'] == 'cifar10':
		opts['training_opts']['dataset']['nb_classes'] = 10
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
		return opts
	if opts['training_opts']['dataset']['method'] == 'voc':
		opts['training_opts']['dataset']['nb_classes'] = 20
		opts['training_opts']['dataset']['input_shape'] = (3, 224, 224)
		return opts
	assert 'Undefined Dataset'


def get_dataset_name(opts):
	return opts['training_opts']['dataset']['method']


def get_merge_flag(opts):
	return opts['model_opts']['param_dict']['data_layer']['merge_flag']['method']


def set_merge_flag(opts, merge_flag):
	opts['model_opts']['param_dict']['data_layer']['merge_flag']['method'] = merge_flag
	return opts


def set_val(fun_name_without_set, val, opts):  ## key should  be compatibale to setter function names
	fun_name = 'set_' + fun_name_without_set
	fun_handle = eval(fun_name)
	opts = fun_handle(opts, val)
	return opts

def get_epoch(opts):
	return opts['training_opts']['epoch_nb']


def get_aug_opts(opts):
	return opts['aug_opts']


def get_lr_sched_family(opts):
	return opts['training_opts']['lr_sched_family']


def set_default_opts_based_on_model_dataset(opts):
	opts['aug_opts']['enable'] = False  # enables augmentation
	opts['aug_opts']['featurewise_center'] = False  # set input mean to 0 over the dataset
	opts['aug_opts']['samplewise_center'] = False  # set each sample mean to 0
	opts['aug_opts']['featurewise_std_normalization'] = False  # divide inputs by std of the dataset
	opts['aug_opts']['samplewise_std_normalization'] = False  # divide each input by its std
	opts['aug_opts']['zca_whitening'] = False# apply ZCA whitening
	opts['aug_opts']['rotation_range'] = 0  # randomly rotate images in the range (degrees, 0 to 180)
	opts['aug_opts']['width_shift_range'] = 0.1  # randomly shift images horizontally (fraction of total width)
	opts['aug_opts']['height_shift_range'] = 0.1  # randomly shift images vertically (fraction of total height)
	opts['aug_opts']['horizontal_flip'] = True  # randomly flip images
	opts['aug_opts']['vertical_flip'] = False

	# Gated Parameters and Activations
	####### ACTIVITY REGULARIZER
	opts['model_opts']['param_dict']['param_expand'] = {}
	opts['model_opts']['param_dict']['param_expand']['rate'] = 1
	# End of Layer Parameters-------------------------------------------------------------------------------------

	opts['optimizer_opts']['lr'] = 1#.1
	opts['optimizer_opts']['momentum'] = .9
	opts['optimizer_opts']['decay'] = 0#1e-6
	opts['optimizer_opts']['nestrov'] = False
	opts['optimizer_opts']['loss']['method'] = KlLoss# 'categorical_crossentropy'

	if opts['training_opts']['dataset']['method'] == 'cifar100':
		opts['training_opts']['dataset']['nb_classes'] = 100
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
	if opts['training_opts']['dataset']['method'] == 'cifar10':
		opts['training_opts']['dataset']['nb_classes'] = 10
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
	if opts['training_opts']['dataset']['method'] == 'voc':
		opts['training_opts']['dataset']['nb_classes'] = 20
		opts['training_opts']['dataset']['input_shape'] = (3, 224, 224)
	opts['training_opts']['samples_per_epoch'] = -1
	opts['training_opts']['batch_size'] = 128
	opts['training_opts']['epoch_nb'] = 265
	opts['training_opts']['metrics'] = ['accuracy', 'mean_absolute_percentage_error', 'cosine_proximity', 'top_k_categorical_accuracy']
	opts['training_opts']['lr_sched_family'] = 'vgg'
	return opts


def default_opt_creator():
	aug_opts = {}

	model_opts = {}
	model_opts['param_dict'] = {}

	optimizer_opts = {}
	optimizer_opts['loss'] = {}

	training_opts = {}
	training_opts['dataset'] = {}
	training_opts['callbacks'] = {}
	training_opts['metrics'] = []
	opts = {'seed': 0, 'experiment_name': '', 'model_opts': model_opts, 'aug_opts': aug_opts, 'training_opts': training_opts, 'optimizer_opts': optimizer_opts}
	return opts
