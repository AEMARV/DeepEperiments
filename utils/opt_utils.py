def set_gate_activation(opts,activation):
	opts['model_opts']['param_dict']['gate_layer']['gate_activation']['method']=activation
	return opts
def get_gate_activation(opts):
	return opts['model_opts']['param_dict']['gate_layer']['gate_activation']['method']
def set_data_activation(opts,activation):
	opts['model_opts']['param_dict']['data_layer']['data_activation']['method']=activation
	return opts
def get_data_activation(opts):
	return opts['model_opts']['param_dict']['data_layer']['data_activation']['method']
def set_gate_activity_regularizer(opts,activity_regularizer_string):
	opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'] = activity_regularizer_string
	return opts
def set_loss(opts,loss_string):
	opts['optimizer_opts']['loss']['method'] = loss_string
	return opts
def set_weight_decay(opts,decay):
	opts['optimizer_opts']['decay']=decay
	return opts
def set_lr(opts,lr):
	opts['optimizer_opts']['lr']=lr
	return opts
def get_model_string(opts):
	return opts['model_opts']['method']
def set_model_string(opts,method_str):
	opts['model_opts']['method']=method_str
	return opts
def set_w_reg(opts,w_reg):
	opts['model_opts']['param_dict']['w_regularizer']['method'] = w_reg
	return opts
def set_w_reg_value(opts,w_reg_value):
	opts['model_opts']['param_dict']['w_regularizer']['value'] = w_reg_value
	return opts
def set_expand_rate(opts,expand_rate):
	opts['model_opts']['param_dict']['param_expand']['rate']=expand_rate
	return opts
def set_stoch(opts,stoch_enable):
	opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch']['method'] = stoch_enable
	return opts
def get_stoc(opts):
	return opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch']['method']
def set_stoc_flip(opts,stoch_enable):
	opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch_flip']['method'] = stoch_enable
	return opts
def get_stoc_flip(opts):
	return opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch_flip']['method']
def get_filter_size(opts):
	return opts['model_opts']['param_dict']['data_layer']['conv_size']['method']
def set_filter_size(opts,size):
	 opts['model_opts']['param_dict']['data_layer']['conv_size']['method']=size
	 return opts
def set_dataset(opts,dataset):
	opts['training_opts']['dataset']['method'] = dataset
	if opts['training_opts']['dataset']['method']=='cifar100':
		opts['training_opts']['dataset']['nb_classes']=100
		opts['training_opts']['dataset']['input_shape'] = (3,32,32)
		return opts
	if opts['training_opts']['dataset']['method']=='cifar10':
		opts['training_opts']['dataset']['nb_classes']=10
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
		return opts
	if opts['training_opts']['dataset']['method']=='voc':
		opts['training_opts']['dataset']['nb_classes']=20
		opts['training_opts']['dataset']['input_shape'] = (3, 224, 224)
		return opts
	assert 'Undefined Dataset'
def get_merge_flag(opts):
	return opts['model_opts']['param_dict']['data_layer']['merge_flag']['method']
def set_merge_flag(opts,merge_flag):
	opts['model_opts']['param_dict']['data_layer']['merge_flag']['method']=merge_flag
	return opts
def set_val(fun_name_without_set,val,opts):## key should  be compatibale to setter function names
	fun_name = 'set_'+fun_name_without_set
	fun_handle = eval(fun_name)
	opts = fun_handle(opts,val)
	return opts

def set_default_opts_based_on_model_dataset(opts):
	opts['aug_opts']['enable'] = False # enables augmentation
	opts['aug_opts']['featurewise_center'] = True  # set input mean to 0 over the dataset
	opts['aug_opts']['samplewise_center'] = False  # set each sample mean to 0
	opts['aug_opts']['featurewise_std_normalization'] = True  # divide inputs by std of the dataset
	opts['aug_opts']['samplewise_std_normalization'] = False  # divide each input by its std
	opts['aug_opts']['zca_whitening'] = True  # apply ZCA whitening
	opts['aug_opts']['rotation_range'] = 0  # randomly rotate images in the range (degrees, 0 to 180)
	opts['aug_opts']['width_shift_range'] = 0  # randomly shift images horizontally (fraction of total width)
	opts['aug_opts']['height_shift_range'] = 0  # randomly shift images vertically (fraction of total height)
	opts['aug_opts']['horizontal_flip'] = False  # randomly flip images
	opts['aug_opts']['vertical_flip'] = False


	# Gated Parameters and Activations
	####### ACTIVITY REGULARIZER
	if opts['model_opts']['method'].find('lenet')==-1:
		opts['model_opts']['param_dict']['gate_layer'] = {}
		opts['model_opts']['param_dict']['gate_layer']['gate_activation'] = {}
		opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer'] = {}
		opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'] = {}
		opts['model_opts']['param_dict']['data_layer'] = {}
		opts['model_opts']['param_dict']['data_layer']['data_activation'] = {}
		# put layer parameters ------------------------------HERE------------------------
		opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['method'] = 'mean_value'
		opts['model_opts']['param_dict']['gate_layer']['gate_activation']['method'] = 'softplus'
		opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch'] = {}
		opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch_flip']={}
		opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch_flip']['method'] = False
		opts['model_opts']['param_dict']['gate_layer']['gate_activation']['stoch']['method'] = True
		opts['model_opts']['param_dict']['data_layer']['data_activation']['method'] = 'elu'
		regularizer_parameter_dict = {}
		if opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['method'] == 'variance_spatial':
			regularizer_parameter_dict = {'alpha': .1}
		if opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['method'] == 'mean_value':
			regularizer_parameter_dict = {'average': .5,'average_reg_weight':[1,1,1,1],'average_reg':[.5,.5,.5,.5]}
		opts['model_opts']['param_dict']['gate_layer']['gate_activity_regularizer']['param_dict'] = regularizer_parameter_dict
	######## Weight Regularizer
	else:
		opts['model_opts']['param_dict']['data_layer'] = {}
		opts['model_opts']['param_dict']['data_layer']['data_activation'] = {}
		opts['model_opts']['param_dict']['data_layer']['data_activation']['method'] = 'relu'
	opts['model_opts']['param_dict']['data_layer']['conv_size']={}
	opts['model_opts']['param_dict']['data_layer']['conv_size']['method']=-1
	# opts['model_opts']['param_dict']['data_layer']['merge_flag']={}
	# opts['model_opts']['param_dict']['data_layer']['merge_flag']['method']=False

	opts['model_opts']['param_dict']['w_regularizer']={}
	opts['model_opts']['param_dict']['w_regularizer']['method'] = None
	opts['model_opts']['param_dict']['w_regularizer']['value'] = 1e-6
	opts['model_opts']['param_dict']['param_expand']={}
	opts['model_opts']['param_dict']['param_expand']['rate']=1
	# End of Layer Parameters-------------------------------------------------------------------------------------

	opts['optimizer_opts']['lr'] = -2
	opts['optimizer_opts']['momentum'] = .9
	opts['optimizer_opts']['decay'] = 1e-6
	opts['optimizer_opts']['nestrov'] = False
	opts['optimizer_opts']['loss']['method'] = 'categorical_crossentropy'

	if opts['training_opts']['dataset']['method']=='cifar100':
		opts['training_opts']['dataset']['nb_classes']=100
		opts['training_opts']['dataset']['input_shape'] = (3,32,32)
	if opts['training_opts']['dataset']['method']=='cifar10':
		opts['training_opts']['dataset']['nb_classes']=10
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
	if opts['training_opts']['dataset']['method']=='voc':
		opts['training_opts']['dataset']['nb_classes']=20
		opts['training_opts']['dataset']['input_shape'] = (3, 224, 224)
	opts['training_opts']['samples_per_epoch'] =-1
	opts['training_opts']['batch_size'] = 128
	opts['training_opts']['epoch_nb']=250
	return opts
def default_opt_creator():
	aug_opts = {}

	model_opts = {}
	model_opts['param_dict'] = {}


	optimizer_opts = {}
	optimizer_opts['loss'] = {}

	training_opts = {}
	training_opts['dataset'] = {}
	opts = {
		'seed'           : 0,
		'experiment_name': '',
		'model_opts'     : model_opts,
		'aug_opts'       : aug_opts,
		'training_opts'  : training_opts,
		'optimizer_opts' : optimizer_opts}
	return opts
