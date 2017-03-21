from __future__ import print_function

import os

import numpy as np
from keras.callbacks import ReduceLROnPlateau,TensorBoard,History,CSVLogger,LearningRateScheduler,ModelCheckpoint
from CallBacks.callbacks import *
from keras.optimizers import SGD
from keras.preprocessing import image
from CallBacks.callback_metric_plot import PlotMetrics
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

cpu_debug = False


def dataset_stats(generator):
	sum = 0
	mean = 0
	for batch_index in np.arange(0, generator.nb_sample / generator.batch_size):
		sum += np.mean(generator.next()[0], axis=(0, 1, 2))
		if np.log2(batch_index + 1) % 1 == 0:
			sum = sum / (float(batch_index + 1) / 2)
			mean = (sum + mean) / 2
			sum = 0
	return mean


def train(model, dataset_abs_path, optimizer, opts,cpu_debug=False):
	use_costum_training = False

	model.compile(loss=opts['optimizer_opts']['loss']['method'],
	              optimizer=optimizer,
	              metrics=['accuracy','categorical_accuracy','mean_absolute_percentage_error','precision','recall'])

	imdg = image.ImageDataGenerator(featurewise_center=True, dim_ordering='th')
	training_abs_path = os.path.join(dataset_abs_path, "training")
	validation_abs_path = os.path.join(dataset_abs_path, "validation")

	# imdg.mean = np.array([[[116]], [[110]], [[102]]])

	train_generator = imdg.flow_from_directory(directory=training_abs_path, target_size=model.input_shape[2:],
	                                           batch_size=opts['training_opts']['batch_size'], seed=0)

	validation_generator = imdg.flow_from_directory(directory=validation_abs_path, target_size=model.input_shape[2:],
	                                                batch_size=opts['training_opts']['batch_size'], seed=0)
	# mean_training = dataset_stats(train_generator)

	if cpu_debug:
		samples_per_epoch_training = 6
		samples_per_epoch_validation = 6
	else:
		samples_per_epoch_training = train_generator.nb_sample
		samples_per_epoch_validation = validation_generator.nb_sample

	if use_costum_training:
		trainer_from_generator(train_generator, opts['training_opts']['batch_size'], model)
		validation_from_generator(validation_generator, opts['training_opts']['batch_size'], model)
	else:
		##TODO: change samples per epoch to trainer_from_generator.nbsamples
		plotter = PlotMetrics(opts)
		experiments_abs_path = plotter.history_holder.dir_abs_path
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
		tensorboard = TensorBoard(log_dir=experiments_abs_path + '/logs', histogram_freq=2, write_images=False)
		csv_logger = CSVLogger(filename=experiments_abs_path + '/training.log', separator=',')
		lr_sched = LearningRateScheduler(lr_random_multiScale)
		callback_list = [plotter, tensorboard, csv_logger]
		if opts['optimizer_opts']['lr'] == -1:
			callback_list = callback_list + [lr_sched]
		# Debugger
		model.fit_generator(generator=train_generator, samples_per_epoch=samples_per_epoch_training,
		                    nb_epoch=opts['training_opts']['epoch_nb'],
		                    validation_data=validation_generator, nb_val_samples=samples_per_epoch_validation,
		                    callbacks=callback_list)


def trainer_from_generator(train_generator, batch_size, model):
	batch_train_total = int(train_generator.nb_sample / batch_size)
	loss = 0
	while train_generator.batch_index < batch_train_total:
		batch = train_generator.next()
		# preds = model.predict_on_batch(batch[0])
		loss_batch = model.train_on_batch(batch[0], batch[1])
		loss += loss_batch[0]
		print(train_generator.batch_index + 1, '/', batch_train_total, ' : Loss: ', loss_batch[0], ' Loss avg: ',
		      loss / train_generator.batch_index, end='\r')
	loss = loss / train_generator.batch_index
	print('Average Loss:', loss)
	train_generator.reset()
def validation_from_generator(validation_generator, batch_size, model):
	batch_val_total = int(validation_generator.nb_sample / batch_size)
	loss = 0
	while validation_generator.batch_index < batch_val_total:
		batch = validation_generator.next()
		loss_batch = model.test_on_batch(batch[0], batch[1])
		loss += loss_batch
	loss = loss / validation_generator.batch_index
	validation_generator.reset

def run_trainer(model,opts,epoch_nb=100, batch_size=64 ,
                dataset_abs_path="/home/student/Documents/Git/Datasets/VOC/Keras", lr=0.001, momentum=.9,
                decay=1e-4, nesterov=True,cpu_debug=False):
	if cpu_debug:
		batch_size = 2
	epoch_nb = opts['epoch_nb']
	batch_size = opts['batch_size']
	lr = opts['lr']
	momentum = opts['momentum']
	decay = opts['decay']
	nestrov = opts['nestrov']
	sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
	train(model, dataset_abs_path, sgd, batch_size=batch_size, epoch_nb=epoch_nb,cpu_debug=cpu_debug,opts=opts)
def lr_random_multiScale(index):
	ler = .01
	# for i in np.arange(1+((index)/20)):
	for i in np.arange(1+(index/80)):
		ler = np.random.random()*ler
		print('learningrate:',ler)
	return ler
def lr_permut(index):
	lr = [1e-2,1e-2,1e-2,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-4,1e-4,1e-4]
	a = np.random.randint(0,3)
	k=(index/30)
	print('rand_lr_value:',k)
	i = np.floor(k)
	if i<0:
		i=0
	if i>=lr.__len__():
		i=lr.__len__()-1
	lr_s = lr[int(i)]
	print('LearningRate:',lr_s)
	return lr_s
def springberg_lr(index):
	S = [200,250,300]
def cifar_trainer(opts,model,optimizer):
	nb_classes=10;

	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	samples_per_epoch=opts['training_opts']['samples_per_epoch']
	if opts['training_opts']['samples_per_epoch']==-1:
		samples_per_epoch=X_train.shape[0]

	model.compile(loss=opts['optimizer_opts']['loss']['method'],
	              optimizer=optimizer,
	              metrics=['accuracy', 'mean_absolute_percentage_error', 'precision', 'recall',
	                       'cosine_proximity', 'top_k_categorical_accuracy', 'fmeasure'])
	# if not data_augmentation:
	# 	print('Not using data augmentation.')
	# 	model.fit(X_train, Y_train,
	# 	          batch_size=batch_size,
	# 	          nb_epoch=epoch_nb,
	# 	          validation_data=(X_test, Y_test),
	# 	          shuffle=True)
	file_name_extension = 'cifar100_zca:' + str(opts['aug_opts']['zca_whitening']) + '_stdn:' + str(
		opts['aug_opts']['featurewise_std_normalization']) + '_mn:' + str(
		opts['aug_opts']['featurewise_center']) + '.npy'
	train_name = 'xtrain_' + file_name_extension
	train_file_path = '/home/student/Documents/Git/DeepEperiments/' + train_name
	test_file_path = '/home/student/Documents/Git/DeepEperiments/' + 'xtest_' + file_name_extension
	if opts['aug_opts']['enable']:
		if not os.path.exists(os.path.abspath(train_file_path)):
			print('Using real-time data augmentation.')
			datagen = ImageDataGenerator(featurewise_center=opts['aug_opts']['featurewise_center'],
			                             featurewise_std_normalization=opts['aug_opts'][
				                             'featurewise_std_normalization'], # divide inputs by std of the dataset
			                             zca_whitening=opts['aug_opts']['zca_whitening'],  # apply ZCA whitening
			                             )
			print('Data Augmentation enabled')
			print(opts['aug_opts'])
			datagen.fit(X_train)
			for i in range(X_train.shape[0]):
				X_train[i] = datagen.standardize(X_train[i])
			for i in range(X_test.shape[0]):
				X_test[i] = datagen.standardize(X_test[i])
			np.save(os.path.abspath(train_file_path), X_train)
			np.save(os.path.abspath(test_file_path), X_test)
		else:
			X_train = np.load(os.path.abspath(train_file_path))
			X_test = np.load(os.path.abspath(test_file_path))


		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
	if opts['aug_opts']['enable']:
		print('Using real-time data augmentation.')
		datagen = ImageDataGenerator(# set input mean to 0 over the dataset
			samplewise_center=opts['aug_opts']['samplewise_center'],  # set each sample mean to 0
			samplewise_std_normalization=opts['aug_opts']['samplewise_std_normalization'],
			# divide each input by its std
			rotation_range=opts['aug_opts']['rotation_range'], # randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=opts['aug_opts']['width_shift_range'],
			# randomly shift images horizontally (fraction of total width)
			height_shift_range=opts['aug_opts']['height_shift_range'],
			# randomly shift images vertically (fraction of total height)
			horizontal_flip=opts['aug_opts']['horizontal_flip'],  # randomly flip images
			vertical_flip=opts['aug_opts']['vertical_flip'])  # randomly flip images
		print('Data Augmentation enabled')
		print(opts['aug_opts'])
	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	else:
		datagen = ImageDataGenerator()
	plotter = PlotMetrics(opts)
	experiments_abs_path = plotter.history_holder.dir_abs_path
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
	tensorboard = TensorboardCostum(log_dir=experiments_abs_path + '/logs', histogram_freq=1, write_graph=False,
	                                write_images=False)
	csv_logger = CSVLogger(filename=experiments_abs_path+'/training.log',separator=',')
	lr_sched = LearningRateScheduler(lr_random_multiScale)
	early_stopping = EarlyStopping('acc', min_delta=.0001, patience=40, mode='max')
	checkpoint = ModelCheckpoint(experiments_abs_path + '/checkpoint', period=10)
	callback_list = [plotter, csv_logger,tensorboard,checkpoint]
	# if opts['optimizer_opts']['lr']==-1:
	# 	callback_list = callback_list+[lr_sched]
	if opts['optimizer_opts']['lr']==-2:
		lr_sched = LearningRateScheduler(lr_permut)
	# if opts['optimizer_opts']['lr'] == -3:
	# lr_sched = LearningNIN()

	callback_list = callback_list+[lr_sched]
	# fit the model on the batches generated by datagen.flow()
	model.fit_generator(
		datagen.flow(X_train, Y_train, batch_size=opts['training_opts']['batch_size'], shuffle=True, seed=opts[
			'seed']),
		samples_per_epoch=samples_per_epoch, nb_epoch=opts['training_opts']['epoch_nb'], callbacks=callback_list,
		validation_data=datagen.flow(X_test, Y_test), nb_val_samples=X_test.shape[0])

def cifar100_trainer(opts, model, optimizer):
	nb_classes = 100;

	(X_train, y_train), (X_test, y_test) = cifar100.load_data()
	Y_train = np_utils.to_categorical(y_train, nb_classes)
	Y_test = np_utils.to_categorical(y_test, nb_classes)
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	samples_per_epoch = opts['training_opts']['samples_per_epoch']
	if opts['training_opts']['samples_per_epoch'] == -1:
		samples_per_epoch = X_train.shape[0]

	model.compile(loss=opts['optimizer_opts']['loss']['method'], optimizer=optimizer,
	              metrics=['accuracy', 'mean_absolute_percentage_error', 'precision',
	                       'recall','cosine_proximity','top_k_categorical_accuracy','fmeasure'])
	# if not data_augmentation:
	# 	print('Not using data augmentation.')
	# 	model.fit(X_train, Y_train,
	# 	          batch_size=batch_size,
	# 	          nb_epoch=epoch_nb,
	# 	          validation_data=(X_test, Y_test),
	# 	          shuffle=True)
	file_name_extension = 'cifar100_zca:'+str(opts['aug_opts']['zca_whitening'])+'_stdn:'+str(opts['aug_opts'][
		                                                                                 'featurewise_std_normalization'])+'_mn:'+str(opts['aug_opts']['featurewise_center'])+'.npy'
	train_name = 'xtrain_'+file_name_extension
	train_file_path = '/home/student/Documents/Git/DeepEperiments/'+train_name
	test_file_path = '/home/student/Documents/Git/DeepEperiments/'+'xtest_'+file_name_extension
	if opts['aug_opts']['enable'] :
		if  not os.path.exists(os.path.abspath(train_file_path)):
			print('Using real-time data augmentation.')
			datagen = ImageDataGenerator(featurewise_center=opts['aug_opts']['featurewise_center'],
				# set input mean to 0 over the dataset
				featurewise_std_normalization=opts['aug_opts']['featurewise_std_normalization'],
				# divide inputs by std of the dataset
				# divide each input by its std
				zca_whitening=opts['aug_opts']['zca_whitening'],  # apply ZCA whitening
				# randomly rotate images in the range (degrees, 0 to 180)
				# randomly shift images horizontally (fraction of total width)
				# randomly shift images vertically (fraction of total height)
			                             )  # randomly flip images
			print('Data Augmentation enabled')
			print(opts['aug_opts'])
			datagen.fit(X_train)
			for i in range(X_train.shape[0]):
				X_train[i] = datagen.standardize(X_train[i])
				print(i)
			for i in range(X_test.shape[0]):
				X_test[i] = datagen.standardize(X_test[i])
			np.save(os.path.abspath(train_file_path),X_train)
			np.save(os.path.abspath(test_file_path),X_test)
		else:
			X_train = np.load(os.path.abspath(train_file_path))
			X_test = np.load(os.path.abspath(test_file_path))


		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
	if opts['aug_opts']['enable']:
		print('Using real-time data augmentation.')
		datagen = ImageDataGenerator(
			# set input mean to 0 over the dataset
			samplewise_center=opts['aug_opts']['samplewise_center'],  # set each sample mean to 0
			samplewise_std_normalization=opts['aug_opts']['samplewise_std_normalization'],
			# divide each input by its std
			rotation_range=opts['aug_opts']['rotation_range'],
			# randomly rotate images in the range (degrees, 0 to 180)
			width_shift_range=opts['aug_opts']['width_shift_range'],
			# randomly shift images horizontally (fraction of total width)
			height_shift_range=opts['aug_opts']['height_shift_range'],
			# randomly shift images vertically (fraction of total height)
			horizontal_flip=opts['aug_opts']['horizontal_flip'],  # randomly flip images
			vertical_flip=opts['aug_opts']['vertical_flip'])  # randomly flip images
		print('Data Augmentation enabled')
		print(opts['aug_opts'])
		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
	else:
		datagen=ImageDataGenerator()
	plotter = PlotMetrics(opts)
	experiments_abs_path = plotter.history_holder.dir_abs_path
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
	tensorboard = TensorboardCostum(log_dir=experiments_abs_path + '/logs', histogram_freq=1, write_graph=False,
	                          write_images=False)
	csv_logger = CSVLogger(filename=experiments_abs_path + '/training.log', separator=',')
	lr_sched = LearningRateScheduler(lr_random_multiScale)
	early_stopping = EarlyStopping('acc',min_delta=.0001,patience=20,mode='max')
	general_callback = GeneralCallback()
	checkpoint  = ModelCheckpoint(experiments_abs_path + '/checkpoint',period=10)
	callback_list = [plotter, csv_logger,general_callback,tensorboard,checkpoint]
	# callback_list = [tensorboard]
	# if opts['optimizer_opts']['lr'] == -1:
	# 	callback_list = callback_list + [lr_sched]
	# if opts['optimizer_opts']['lr']==-2:
	lr_sched = LearningRateScheduler(lr_permut)
	# 	callback_list = callback_list+[lr_sched]
	# lr_sched = LearningNIN()
	callback_list = callback_list+[lr_sched]
	# fit the model on the batches generated by datagen.flow()
	model.fit_generator(
		datagen.flow(X_train, Y_train, batch_size=opts['training_opts']['batch_size'], shuffle=True,
		             seed=opts['seed']),samples_per_epoch=samples_per_epoch,
		nb_epoch=opts['training_opts']['epoch_nb'],  callbacks=callback_list,validation_data=(X_test,Y_test))
	# else:
	# 	model.fit(X_train, Y_train, batch_size=opts['training_opts']['batch_size'],
	# 	          nb_epoch=opts['training_opts']['epoch_nb'], validation_data=(X_test, Y_test), shuffle=True,
	# 	          callbacks=callback_list)
# if __name__ == '__main__':
# 	aug_opts={}
# 	aug_opts['featurewise_center'] = False  # set input mean to 0 over the dataset
# 	aug_opts['samplewise_center'] = False  # set each sample mean to 0
# 	aug_opts['featurewise_std_normalization'] = False  # divide inputs by std of the dataset
# 	aug_opts['samplewise_std_normalization'] = False  # divide each input by its std
# 	aug_opts['zca_whitening'] = False  # apply ZCA whitening
# 	aug_opts['rotation_range'] = 0  # randomly rotate images in the range (degrees, 0 to 180)
# 	aug_opts['width_shift_range'] = 0.1  # randomly shift images horizontally (fraction of total width)
# 	aug_opts['height_shift_range'] = 0.1  # randomly shift images vertically (fraction of total height)
# 	aug_opts['horizontal_flip'] = True  # randomly flip images
# 	aug_opts['vertical_flip'] = False
#
# 	opts = {'experiment_name':'Baseline_lenet_random_lr','lr':0.01,'momentum':.9,'decay':1e-6,
# 	        'nestrov':True,'batch_size':64,'epoch_nb':100,'act_regul_var_alpha':.7,
# 	        'samples_per_epoch':-1,'dataset':'cifar10','model':'lenet','augmentation_opts':aug_opts}
# 	opts['experiment_name']=raw_input('experiment name?')
# 	opts['augmentation']=False
# 	opts['seed']=0
# 	epoch_nb = opts['epoch_nb']
# 	batch_size = opts['batch_size']
# 	lr = opts['lr']
# 	momentum=opts['momentum']
# 	decay = opts['decay']
# 	nestrov = opts['nestrov']
# 	dataset = opts['dataset']
# 	opts['experiment_name'] = opts['experiment_name']+'_'+opts['dataset']+'_'+ 'epochs_'+str(epoch_nb)
# 	# lr  = np.random.rand(10,1)
# 	 # = np.random.rand(10,1)*np.random.rand(10,1)
# 	# model = VGG16(weights=None)
# 	# model = gate_net.gated_lenet(depth = 5,nb_filter=4,filter_size=3,opts=opts,input_shape=(3,32,32))
# 	model = lenet.lenet_model()
# 	sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nestrov)
# 	if dataset == 'voc':
# 		dataset_abs_path = "/home/student/Documents/Git/Datasets/VOC/Keras"
# 		train(model, dataset_abs_path, sgd, batch_size=batch_size, epoch_nb=epoch_nb,cpu_debug=False,opts=opts)
# 	if dataset=='cifar10':
# 		cifar_trainer(opts)




