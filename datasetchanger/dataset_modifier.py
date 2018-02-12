from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imread
import tensorflow as tf
from scipy.optimize import fmin_l_bfgs_b
import scipy.ndimage
from keras.objectives import categorical_crossentropy
import keras.backend as K
def corrupt_dataset_model(model:Model):
	prediction_tensor = model.output
	labels_tensor = K.placeholder(K.int_shape(prediction_tensor))
	input_tensor = model.input
	prediction_tensor = K.clip(prediction_tensor,K.epsilon(),1-K.epsilon())
	q = K.log(prediction_tensor)
	loss = K.categorical_crossentropy(prediction_tensor,labels_tensor)
	grads = K.gradients(loss,input_tensor)
	res = input_tensor+.1*grads
	fun = K.function([input_tensor,labels_tensor],[res])
	return fun ,input_tensor,labels_tensor
def corrupt_dataset(model: Model, dataset_imgs: np.ndarray, dataset_labels: np.ndarray):
	fun,input_tensor,labels_tensor = corrupt_dataset_model(model)
	session = K.get_session()
	tensors = [input_tensor,labels_tensor]
	batch_val = [dataset_imgs,dataset_labels]
	feed_dict = dict(list(zip(tensors, batch_val)))
	res = session.run([input_tensor,labels_tensor],feed_dict=feed_dict)
	a = imshow(res[0])
	plt.show(a)
	input('a')

def get_crupted_tensor(model_input, model_prediction, labels_tensor):
	# labels_tensor = K.placeholder(K.int_shape(model_prediction))
	loss = K.categorical_crossentropy(model_prediction, labels_tensor)
	grads = K.gradients(loss, model_input)
	res = model_input + grads[0]
	return res
def get_crupted_funciton(model_input,model_prediction,labels_tensor):
	# labels_tensor = K.placeholder(K.int_shape(model_prediction))
	loss = K.categorical_crossentropy(model_prediction, labels_tensor)
	grads = K.gradients(loss, model_input)
	res = model_input +  grads[0]
	return K.function([model_input,labels_tensor],outputs=[grads[0],res])