from myTrainer import *
from Models.binary_net import gatenet_binary,gatenet_binary_merged
from utils.opt_utils import *
from Models.vgg16_zisserman import VGG16
from Models.lenet_amir import lenet_amir_model

import time
def find_key_value_to_str_recursive(dictionary,parent_dict_name,method_name_exclusive_list):
	result_str = ''
	for i in dictionary.iteritems():
		if (i[0]=='method' or i[0]=='rate') and method_name_exclusive_list.__contains__(parent_dict_name):
			result_str+='#'+(parent_dict_name+':'+str(i[1]))
		if type(i[1])==dict:
			result_str+=find_key_value_to_str_recursive(i[1],i[0],method_name_exclusive_list)
	return result_str
def grid_search(opts,experiment_name=None,model_str=None,dataset_str=None):
	if experiment_name==None:
		opts['experiment_name'] = raw_input('experiment_name?')
		experiment_name=opts['experiment_name']
	else:
		opts['experiment_name'] = experiment_name
	opts['experiment_tag']=experiment_name+'/'+dataset_str+'/'+model_str
	if model_str==None:model_str = raw_input('model?')
	if dataset_str==None:opts['training_opts']['dataset']['method'] = raw_input('dataset?')
	else:opts['training_opts']['dataset']['method'] = dataset_str
	opts = set_model_string(opts,model_str)
	# activation_set = {'softplus','elu','softsign','relu','tanh','softmax','linear'}
	# loss_set = {'kld','poisson','mape','categorical_crossentropy'}
	if opts['training_opts']['dataset']['method']=='cifar100':
		opts['training_opts']['dataset']['nb_classes']=100
		opts['training_opts']['dataset']['input_shape'] = (3,32,32)
	if opts['training_opts']['dataset']['method']=='cifar10':
		opts['training_opts']['dataset']['nb_classes']=10
		opts['training_opts']['dataset']['input_shape'] = (3, 32, 32)
	if opts['training_opts']['dataset']['method']=='voc':
		opts['training_opts']['dataset']['nb_classes']=20
		opts['training_opts']['dataset']['input_shape'] = (3, 224, 224)
	opts = set_default_opts_based_on_model_dataset(opts)


	gate_activation_set = ['relu','softplus']
	gate_stoch_enable =[False,True]
	data_activation_set = [None]
	loss_set = {'categorical_crossentropy'}
	lenet_activation = ['elu']
	filter_size_set = [-1]
	lr = [-2]
	w_reg = {None,'l1'}
	w_reg_value = {1e-6}
	param_expand = [1,2,4] ## in gated all the params are devided by two because we have two layers per channel so
	# this ratio can be compared for number of parameters e.g if in lennet param_expand=1 and in gated param_expand=1
	#  means they have the same number of parameters
	new_opts = [{'gate_activation':['softplus']},{'data_activation':[None]},{'loss':['categorical_crossentropy']}]
	model_str = get_model_string(opts)
	if not str(model_str).find('gatenet') == -1:
		for gate_activation in gate_activation_set:
			for data_activation in data_activation_set:
				for loss_objective in loss_set:
					for lr_sel in lr:
						for w_reg_sel in w_reg:
							for w_reg_value_sel in w_reg_value:
								for param_expand_sel in param_expand:
									for gate_stoch in gate_stoch_enable:
										for f_size in filter_size_set:
											if gate_activation=='relu' and gate_stoch is True:
												break
											if gate_activation == 'softplus' and gate_stoch is False:
												break
											opts=set_gate_activation(opts=opts,activation=gate_activation)
											opts=set_data_activation(opts=opts,activation=data_activation)
											opts=set_loss(opts=opts,loss_string=loss_objective)
											opts=set_lr(opts=opts,lr=lr_sel)
											opts= set_w_reg(opts,w_reg_sel)
											opts = set_w_reg_value(opts,w_reg_value_sel)
											opts = set_expand_rate(opts,param_expand_sel)
											opts = set_stoch(opts,gate_stoch)
											opts = set_filter_size(opts,f_size)
											# gated = gate_net.gated_lenet(opts=opts, input_shape=opts['training_opts']['dataset'][
											# 	'input_shape'], nb_classes=opts['training_opts']['dataset']['nb_classes'])
											# print 'gated run'
											# d_gated = gate_net.dropout_gated_lenet(opts=opts, input_shape=opts['training_opts']['dataset'][
											# 	'input_shape'], nb_classes=opts['training_opts']['dataset']['nb_classes'])
											# binary_bit_generator(opts=opts, input_shape=opts[
											# 	'training_opts']['dataset'][
											# 	'input_shape'], nb_classes=opts['training_opts']['dataset']['nb_classes'])
											gatenet_amir_model = gate_net.gatenet_amir(opts=opts, input_shape=opts[
												'training_opts']['dataset'][
												'input_shape'], nb_classes=opts['training_opts']['dataset']['nb_classes'])
											gatenet_binary_model = gatenet_binary(opts=opts, input_shape=
											opts['training_opts']['dataset']['input_shape'], nb_classes=
											                                     opts['training_opts']['dataset'][
												                                     'nb_classes'])
											gatenet_binary_merged_model = gatenet_binary_merged(opts=opts, input_shape=
											opts['training_opts']['dataset']['input_shape'], nb_classes=
											                                     opts['training_opts']['dataset'][
												                                     'nb_classes'])
											model = eval(model_str)
											model.summary()
											print 'gate_activation:', gate_activation
											print 'data_activation:', data_activation
											print 'loss:', loss_objective
											print 'lr:', lr_sel
											wrapper_gated(model,opts,experiment_name)
	else:
		for loss_objective in loss_set:
			for lr_sel in lr:
				for w_reg_sel in w_reg:
					for w_reg_value_sel in w_reg_value:
						for param_expand_sel in param_expand:
							for actvation in lenet_activation:
								for f_size in filter_size_set:
									print lr_sel
									print loss_objective
									opts = set_loss(opts=opts, loss_string=loss_objective)
									opts = set_lr(opts=opts, lr=lr_sel)
									opts = set_w_reg(opts,w_reg_sel)
									opts = set_w_reg_value(opts,w_reg_value_sel)
									opts = set_expand_rate(opts,param_expand_sel)
									opts = set_data_activation(opts,activation=actvation)
									opts = set_filter_size(opts, f_size)
									print 'loss:', loss_objective
									lenett = lenet.lenet_model(opts=opts,nb_classes=opts['training_opts']['dataset']['nb_classes'])
									lenet_amir = lenet_amir_model(opts=opts,nb_classes=opts['training_opts']['dataset']['nb_classes'])
									if (model_str == 'gated'):
										gated = gate_net.gated_lenet(opts=opts,
										                             input_shape=opts['training_opts']['dataset']['input_shape'],
										                             nb_classes=opts['training_opts']['dataset']['nb_classes'])
										print 'gated run'
									vgg16 = VGG16(weights=None)

									model = eval(model_str)
									model.summary()
									wrapper_gated(model,opts,experiment_name)


def wrapper_gated(model,opts,experiment_name):
	opts['experiment_name']= experiment_name
	method_names = find_key_value_to_str_recursive(opts,'',{'param_expand','gate_activation','stoch'})
	opts['experiment_name'] =method_names+'#param_count:'+str(model.count_params())
	sgd = SGD(lr=opts['optimizer_opts']['lr'], momentum=opts['optimizer_opts']['momentum'],
	          decay=opts['optimizer_opts']['decay'], nesterov=opts['optimizer_opts']['nestrov'])

	if opts['training_opts']['dataset']['method'] == 'voc':
		dataset_abs_path = "/home/student/Documents/Git/Datasets/VOC/Keras"
		train(model, dataset_abs_path, sgd, cpu_debug=False, opts=opts)
	if opts['training_opts']['dataset']['method'] == 'cifar10':
		cifar_trainer(opts, model, optimizer=sgd)
	if opts['training_opts']['dataset']['method'] == 'cifar100':
		cifar100_trainer(opts, model, optimizer=sgd)
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



if __name__ == '__main__':
	#gatenet_binary_merged_model lenet_amir ,gatenet_binary_model
	models= ['gatenet_binary_merged_model']
	datasets=['cifar10','cifar100']
	experiment_name = 'BiRELU Baseline-Merged'+time.strftime('%b%d')

	for dataset_str in datasets:
		for model_str in models:
			options={}
			options = default_opt_creator()
			options['description'] = 'Testing ReLU activation'
			grid_search(options,experiment_name=experiment_name,model_str=model_str,dataset_str=dataset_str)
# method_names = find_key_value_to_str_recursive(opts,'')

# gated = gate_net.gated_lenet(depth=5, nb_filter=4, filter_size=3, opts=opts, input_shape=(3, 32,
#                                                                                                          32))
# lenet = lenet.lenet_model()
# opts['experiment_name'] = raw_input('experiment name?')
# model = input('model?')
# opts['experiment_name'] = opts['experiment_name']+method_names
# sgd = SGD(lr=opts['optimizer_opts']['lr'], momentum=opts['optimizer_opts']['momentum'],
#           decay=opts['optimizer_opts']['decay'], nesterov=opts['optimizer_opts']['nestrov'])
#
# if opts['training_opts']['dataset'] == 'voc':
# 	dataset_abs_path = "/home/student/Documents/Git/Datasets/VOC/Keras"
# 	train(model, dataset_abs_path, sgd, batch_size=opts['training_opts']['batch_size'],
# 	      epoch_nb=opts['training_opts']['epoch_nb'], cpu_debug=False, opts=opts)
# if opts['training_opts']['dataset'] == 'cifar10':
# 	cifar_trainer(opts, model, optimizer=sgd)
