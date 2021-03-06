import keras
from keras import optimizers
from keras.layers import Input

from modeldatabase.Binary_models.model_db import get_model_from_db, get_model_dict_from_db
from sci_utils.utils import *
from utils.gen_utils import *
from utils.modelutils.layers.conversation import KnowledgeAvoid
from utils.opt_utils import *
from utils.trainingutils.training_phases_utils import *
import matplotlib.pyplot as plt
from utils.modelutils import model_modification_utils
from modeldatabase.Binary_models import model_constructor_utils

def check_model_list(model_list, datasets):
	opts = default_opt_creator()
	for dataset_str in datasets:
		for model in model_list:
			set_dataset(opts, dataset=dataset_str)
			set_default_opts_based_on_model_dataset(opts)
			set_model_string(opts, model)
			get_model_from_db(model, opts)


def imshow_compat(img_batch):
	if len(np.shape(img_batch)) == 4:
		im_res = np.transpose(img_batch, [0, 2, 3, 1])
	else:
		im_res = np.transpose(img_batch, [1, 2, 0])
	return im_res


def keras_compat(img_batch):
	return np.transpose(img_batch, [0, 3, 1, 2])


def print_metrics(results, metrics):
	a = dict(list(zip(metrics, results)))
	for key in a.keys():
		if key in ['acc', 'loss']:
			print('\n', key, '\t', a[key])


def sigmoid(x):
	return 1 / (1 + np.exp(-x))
def construct_model(model_str, opts):
	model_dict = get_model_dict_from_db(model_str, opts)
	# training_model = model_dict['model']
	model_input = model_dict['in']
	predictions = model_dict['out']
	label_tensor = Input(shape=(10,), name='labels')
	grads = KnowledgeAvoid()([model_input, label_tensor, predictions])
	# corrupt_dataset_fun = dataset_modifier.get_crupted_funciton(model_input, predictions, label_tensor)
	training_model = Model([model_input], [predictions])
	corrupt_model = Model([model_input, label_tensor], [grads])
	return training_model, corrupt_model


if __name__ == '__main__':
	# gatenet_binary_merged_model lenet_amir ,gatenet_binary_model
	# with tf.get_default_graph() as graph:
	model_str = 'nin_baseline'
	dataset_str = 'cifar10'
	#
	experiment_name = get_experiment_name_prompt()
	# check_model_list(models, datasets)
	print((keras.__version__))
	graph = tf.get_default_graph()
	print(100 * '*', 3 * '\n', model_str, '\n', dataset_str, 3 * '\n', 100 * '*')
	opts = default_opt_creator()
	opts['experiment_name'] = experiment_name
	opts['experiment_tag'] = experiment_name + '/' + dataset_str + '/' + model_str
	set_dataset(opts, dataset=dataset_str)
	opts = set_model_string(opts, model_str)
	opts = set_default_opts_based_on_model_dataset(opts)
	input_shape = opts['training_opts']['dataset']['input_shape']
	nb_class = opts['training_opts']['dataset']['nb_classes']
	optimizer_a = optimizers.SGD(lr=opts['optimizer_opts']['lr'], momentum=opts['optimizer_opts']['momentum'], decay=opts['optimizer_opts']['decay'],
	                             nesterov=opts['optimizer_opts']['nestrov'])
	# optimizer = optimizers.Adadelta()
	""" MODEL PREPARE """

	# model_dict = get_model_dict_from_db(model_str, opts)
	# training_model = model_dict['model']
	# model_input = model_dict['in']
	# predictions = model_dict['out']
	# label_tensor = Input(shape=(10,), name='labels')
	# [corrupt_dataset, grads] = KnowledgeAvoid()([model_input, label_tensor, predictions])
	# # corrupt_dataset_fun = dataset_modifier.get_crupted_funciton(model_input, predictions, label_tensor)
	# training_model_a = Model([model_input], [predictions])
	training_model_a, corrupt_model_a = construct_model(model_str, opts)
	# training_model_b, corrupt_model_b = construct_model(model_str, opts)
	# training_model_b = Model([model_input], [predictions])
	# corrupt_model_a = Model([model_input, label_tensor], [corrupt_dataset, grads])
	# corrupt_model_a = Model([model_input, label_tensor], [corrupt_dataset, grads])
	# training_model.summary()
	model_modification_utils.load_weights_by_block_index_list(training_model_a, [1, 2, 3, 4, 5, 6, 7, 8, 9], os.path.join(
		global_constant_var.get_experimentcase_abs_path(experiment_name, dataset_str, 'nin_baseline'), 'checkpoint'),
	                                                          model_constructor_utils.CONVSH_NAME)
	training_model_a.compile(loss=opt_utils.get_loss(opts), optimizer=optimizer_a, metrics=opt_utils.get_metrics(opts))
	# training_model_b.compile(loss=opt_utils.get_loss(opts), optimizer=optimizer_b, metrics=opt_utils.get_metrics(opts))
	graph_train = tf.get_default_graph()
	method_names = find_key_value_to_str_recursive(opts, '', {'param_expand'})
	opts['experiment_name'] = method_names
	# LOAD DATA
	(data_train, label_train), (data_test, label_test) = load_data(dataset_str, opts)
	data_train, data_test = preprocess_data_phase(opts, data_train, data_test)
	data_gen = data_augmentation_phase(opts)
	data_train = (np.random.randn(500,1,128,128)+.5)/2
	data_train = np.concatenate(3*[data_train],axis=1)
	# data_train = (np.arange(50000, 3, 32, 32) + .5) / 2
	# data_train = np.zeros([10000,3,32,32])
	CIFAR10_LABELS_LIST = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	# data_train=normalize_img(data_train)
	# COLLECT CALLBACKS
	callback_list = collect_callbacks(opts)
	# TRAIN
	save_scientist_model('leader',training_model_a)
	samples_per_epoch = data_train.shape[0] if opts['training_opts']['samples_per_epoch'] == -1 else opts['training_opts']['samples_per_epoch']
	i = 0
	sci_number = 1
	label_num = np.argmax(label_train,axis=1)
	vanil_dataset_sliced,label_sliced = slice_dataset(data_train,label_train,sci_number)
	# Prepare Noise Image Dataset
	prepared_dataset=[]
	for sci_idx in np.arange(sci_number):
		print('scientist_{}'.format(sci_idx))
		transformed_dataset = transform_dataset(corrupt_model_a, vanil_dataset_sliced[sci_idx][0:30], label_sliced[sci_idx][0:30], 1000, i,
		                                        training_model_a,
		                                        optimize=True)
		np.save('./noise_dataset',transformed_dataset)
		np.save('./noise_label', label_sliced[sci_idx])

		prepared_dataset+= [transformed_dataset]
		for i in np.arange(30):
			# keras.utils.to_categorical()
			plt.imsave('scimage/noise_signiture_sci_{}_imagh_{}_class_{}.png'.format(sci_idx,i,CIFAR10_LABELS_LIST[label_num[i]]), imshow_compat(
				normalize_img(transformed_dataset))[i])

	prepared_dataset = np.concatenate(prepared_dataset)

	# data_train = np.load('./noise_dataset.npy')
	# label_train = np.load('./noise_label.npy')
	# train vanil_model on prepared signature dataset
			# vanil_dataset_sliced = prepared_dataset

		# prepared_dataset = np.concatenate(prepared_dataset_list)
		# prepared_label = np.concatenate(prepared_label_list)
		# del prepared_label_list
		# del prepared_dataset_list
		# del transformed_dataset
		# Train

	# load_scientist_model('leader', training_model_a)
	# training_model_a.fit_generator(
	# 	data_gen.flow(data_train[1:1000], label_train, batch_size=opts['training_opts']['batch_size'], shuffle=True),
	# 	samples_per_epoch=128, nb_epoch=10, callbacks=callback_list, validation_data=(data_test, label_test))
	# results = training_model_a.evaluate(data_test,label_test)
	# print_metrics(results,training_model_a.metrics_names)
	# save_scientist_model('leader', training_model_a)



		# z_a  = corrupt_model_a.predict([data_train_slice_a, label_train_slice_a])



		# grads_batch_a = z_a[0]
		# grads_batch_b = z_b[0]
		# data_train_slice_a = np.concatenate((original_data_a, normalize_img(data_train_slice_a - .2 * grads_batch_a)))
		# data_train_slice_b = np.concatenate((original_data_b, normalize_img(data_train_slice_b + .2 * grads_batch_b)))
		# plt.imshow(imshow_compat(data_train_slice_normalize[0]))
		# plt.imsave('scimage/output_normalized_{}.png'.format(i), imshow_compat(data_train_slice_normalize)[0])
		# plt.imsave('scimage/output_{}.png'.format(i), imshow_compat(data_train_slice)[0])
		# plt.imsave('scimage/dif_{}'.format(i), imshow_compat(normalize_img(np.abs(data_train - data_train_slice)[0])))
		# plt.imsave('scimage/grads_{}.png'.format(i), imshow_compat(normalize_img(grads_batch))[0])
		# plt.imsave('scimage/original_{}.png'.format(i), imshow_compat(normalize_img(data_train))[0])
