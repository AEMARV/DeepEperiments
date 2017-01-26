from ResultManager.history_holder import test_HH
def default_opt_creator():
	aug_opts = {}
	aug_opts['enable'] = False  # enables augmentation
	aug_opts['featurewise_center'] = False  # set input mean to 0 over the dataset
	aug_opts['samplewise_center'] = False  # set each sample mean to 0
	aug_opts['featurewise_std_normalization'] = False  # divide inputs by std of the dataset
	aug_opts['samplewise_std_normalization'] = False  # divide each input by its std
	aug_opts['zca_whitening'] = False  # apply ZCA whitening
	aug_opts['rotation_range'] = 0  # randomly rotate images in the range (degrees, 0 to 180)
	aug_opts['width_shift_range'] = 0.1  # randomly shift images horizontally (fraction of total width)
	aug_opts['height_shift_range'] = 0.1  # randomly shift images vertically (fraction of total height)
	aug_opts['horizontal_flip'] = True  # randomly flip images
	aug_opts['vertical_flip'] = False

	model_opts = {}
	model_opts['method'] = 'gated'
	model_opts['param_dict'] = {}

	# Gated Parameters and Activations
	if model_opts['method'] == 'gated':
		model_opts['param_dict']['gate_layer'] = {}
		model_opts['param_dict']['gate_layer']['gate_activation'] = {}
		model_opts['param_dict']['gate_layer']['gate_activity_regularizer'] = {}
		model_opts['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'] = {}
		model_opts['param_dict']['data_layer'] = {}
		model_opts['param_dict']['data_layer']['data_activation'] = {}
		# put layer parameters ------------------------------HERE------------------------
		model_opts['param_dict']['gate_layer']['gate_activity_regularizer']['method'] = 'None'
		model_opts['param_dict']['gate_layer']['gate_activation']['method'] = 'softplus'
		model_opts['param_dict']['data_layer']['data_activation']['method'] = 'elu'
	regularizer_parameter_dict = {}
	if model_opts['param_dict']['gate_layer']['gate_activity_regularizer']['method'] == 'variance_spatial':
		regularizer_parameter_dict = {'alpha': .1}
	if model_opts['param_dict']['gate_layer']['gate_activity_regularizer']['method'] == 'mean_value':
		regularizer_parameter_dict = {'average': .5}
	# End of Layer Parameters-------------------------------------------------------------------------------------
	model_opts['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'] = regularizer_parameter_dict

	optimizer_opts = {}
	optimizer_opts['loss'] = {}
	optimizer_opts['lr'] = 0.01
	optimizer_opts['momentum'] = .9
	optimizer_opts['decay'] = 1e-6
	optimizer_opts['nestrov'] = False
	optimizer_opts['loss']['method'] = 'categorical_crossentropy'

	training_opts = {}
	training_opts['dataset'] = 'cifar10'
	training_opts['samples_per_epoch'] = -1
	training_opts['batch_size'] = 64
	training_opts['epoch_nb'] = 100
	opts = {
		'seed'           : 0,
		'experiment_name': '',
		'model_opts'     : model_opts,
		'aug_opts'       : aug_opts,
		'training_opts'  : training_opts,
		'optimizer_opts' : optimizer_opts}
	return opts
if __name__ == '__main__':
	opts = default_opt_creator()
	test_HH(opts)