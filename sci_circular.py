import keras
from keras import optimizers
from keras.layers import Input

from modeldatabase.Binary_models.model_db import get_model_from_db, get_model_dict_from_db
from sci_utils.utils import *
from utils.gen_utils import *
from utils.modelutils.layers.conversation import KnowledgeAvoid
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
	model_str = 'simplenn_van'
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
	# model_modification_utils.load_weights_by_block_index_list(training_model_a, [1, 2, 3, 4, 5, 6, 7, 8, 9], os.path.join(
	# 	global_constant_var.get_experimentcase_abs_path(experiment_name, dataset_str, 'simplenn_BE'), 'checkpoint'),
	#                                                           model_constructor_utils.CONVSH_NAME)
	training_model_a.compile(loss=opt_utils.get_loss(opts), optimizer=optimizer_a, metrics=opt_utils.get_metrics(opts))
	# training_model_b.compile(loss=opt_utils.get_loss(opts), optimizer=optimizer_b, metrics=opt_utils.get_metrics(opts))
	graph_train = tf.get_default_graph()
	# corrupt_model.compile(optimizer=optimizer,loss=opt_utils.get_loss(opts))
	# graph = tf.get_default_graph()
	method_names = find_key_value_to_str_recursive(opts, '', {'param_expand'})
	opts['experiment_name'] = method_names
	# LOAD DATA
	(data_train, label_train), (data_test, label_test) = load_data(dataset_str, opts)
	data_train, data_test = preprocess_data_phase(opts, data_train, data_test)
	data_gen = data_augmentation_phase(opts)
	# COLLECT CALLBACKS
	callback_list = collect_callbacks(opts)
	# TRAIN
	samples_per_epoch = data_train.shape[0] if opts['training_opts']['samples_per_epoch'] == -1 else opts['training_opts']['samples_per_epoch']
	i = 0
	# im_original = data_train[0]
	# im_original = np.transpose(im_original, [1, 2, 0])
	# b=plt.imshow(im_original)
	# plt.show(b)
	sci_number = 5
	# data_train_slice = data_test[start:end]
	# label_train_slice = label_test[start:end]
	# vanil_dataset_sliced, label_sliced = slice_dataset(data_train, label_train, sci_number)
	vanil_dataset_sliced = data_train
	label_sliced = label_train
	# training_dataset = vanil_dataset_sliced
	create_weight_file(training_model_a, sci_number)
	for i in np.arange(400):
		for sci_idx_main in np.arange(sci_number):
			print('scientist_{}'.format(sci_idx_main))
			# prepare dataset
			# prepared_dataset_list = [vanil_dataset_sliced[sci_idx_main]]
			# prepared_label_list = [label_sliced[sci_idx_main]]
			sci_idx_partner = (sci_idx_main - 1) % sci_number
			if not (i == 0 and sci_idx_main == 0):
				load_scientist_model(sci_idx_partner, training_model_a)
				transformed_dataset = transform_dataset(model=corrupt_model_a, vanila_dataset=vanil_dataset_sliced, labels=label_sliced,
				                                        iterations=8, epoch=i,
				                                        eval_model=training_model_a,
				                                        optimize=True)
				prepared_dataset = np.concatenate([transformed_dataset,data_train],axis=0)

				# vanil_dataset_sliced = prepared_dataset
				prepared_label = np.concatenate([label_sliced,label_sliced],axis=0)
			else:
				prepared_dataset = data_train
				prepared_label = label_sliced

			# prepared_dataset = np.concatenate(prepared_dataset_list)
			# prepared_label = np.concatenate(prepared_label_list)
			# del prepared_label_list
			# del prepared_dataset_list
			# del transformed_dataset
			# Train
			load_scientist_model(sci_idx_main, training_model_a)
			training_model_a.fit_generator(
				data_gen.flow(prepared_dataset, prepared_label, batch_size=opts['training_opts']['batch_size'], shuffle=True),
				samples_per_epoch=samples_per_epoch, nb_epoch=10, callbacks=callback_list, validation_data=(data_test, label_test))
			save_scientist_model(sci_idx_main, training_model_a)



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
