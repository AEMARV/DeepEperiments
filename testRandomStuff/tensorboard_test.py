import numpy as np
from keras.layers import Activation,Dense,Convolution2D,Flatten
from keras.models import Sequential
from keras.callbacks import TensorBoard
import keras.backend as K
if __name__ == '__main__':

	data = np.random.random((1000, 3,10,10))
	labels = np.random.randint(2, size=(1000, 1))

	val_data = np.random.random((1000, 3,10,10))
	val_labels = np.random.randint(2, size=(1000, 1))

	K.clear_session()
	model = Sequential()
	model.add(Convolution2D(30,3,3,input_shape=(3,10,10)))
	model.add(Convolution2D(30, 3, 3, input_shape=(3, 10, 10)))
	model.add(Convolution2D(30, 3, 3, input_shape=(3, 10, 10)))
	model.add(Convolution2D(30, 3, 3, input_shape=(3, 10, 10)))
	model.add(Flatten())
	model.add(Dense(10))

	model.add(Dense(10))

	model.add(Dense(10))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(optimizer='sgd', loss='binary_crossentropy')

	model.fit(data, labels, validation_data=(val_data, val_labels), nb_epoch=10, batch_size=32,
	          callbacks=[TensorBoard(histogram_freq=1,write_graph=False,write_images=False)])