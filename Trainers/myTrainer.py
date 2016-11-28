from __future__ import print_function
import os
from keras.callbacks import TensorBoard
import numpy as np
from sklearn import metrics
from keras import backend as K
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import fbeta_score
from keras.preprocessing import image
from Models.vgg16_zisserman import VGG16
from Metrics import myMetrics
from CallBacks.plotMetrics import PlotMetrics

cpu_debug=True

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


def train(model, dataset_abs_path, optimizer, batch_size, epoch_nb):
	use_costum_training = False

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
	model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['acc','categorical_crossentropy'])

	imdg = image.ImageDataGenerator(featurewise_center=True, dim_ordering='th')
	training_abs_path = os.path.join(dataset_abs_path, "training")
	validation_abs_path = os.path.join(dataset_abs_path, "validation")

	imdg.mean = np.array([[[116]], [[110]], [[102]]])
	if cpu_debug:
		batch_size=2
	train_generator = imdg.flow_from_directory(directory=training_abs_path, target_size=model.input_shape[2:],
	                                           batch_size=batch_size, seed=0)

	validation_generator = imdg.flow_from_directory(directory=validation_abs_path, target_size=model.input_shape[2:],
	                                                batch_size=batch_size, seed=0)
	# mean_training = dataset_stats(train_generator)

	if cpu_debug:
		samples_per_epoch_training = 6
		samples_per_epoch_validation = 6
	else:
		samples_per_epoch_training = train_generator.nb_sample
		samples_per_epoch_validation = validation_generator.nb_sample

	if use_costum_training:
		trainer_from_generator(train_generator, batch_size, model)
		validation_from_generator(validation_generator, batch_size, model)
	else:
		##TODO: change samples per epoch to trainer_from_generator.nbsamples
		model.fit_generator(generator=train_generator, samples_per_epoch=samples_per_epoch_training,
		                    nb_epoch=epoch_nb,
		                    validation_data=validation_generator, nb_val_samples=samples_per_epoch_validation,
		                    callbacks=[reduce_lr,PlotMetrics()])


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

if __name__ == '__main__':
	epoch_nb = 100
	batch_size = 64
	model = VGG16(weights=None)
	dataset_abs_path = "/home/student/Documents/Git/Datasets/VOC/Keras"
	sgd = SGD(lr=0.001, momentum=.9, decay=1e-4, nesterov=True)
	train(model, dataset_abs_path, sgd, batch_size=batch_size, epoch_nb=epoch_nb)
