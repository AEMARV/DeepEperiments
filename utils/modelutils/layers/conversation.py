from keras.layers import Layer
from utils.modelutils.activations.activations import *
class KnowledgeAvoid(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = False
		super(KnowledgeAvoid, self).__init__(**kwargs)

	# def compute_output_shape(self, input_shape):
	# 	return (input_shape[0],input_shape[0])
	def compute_output_shape(self, input_shape):
		return input_shape[0]
	def compute_mask(self, input, input_mask=None):
		return None

	def call(self, x, mask=None):
		model_input = x[0]
		model_input = sigmoid(model_input)
		labels_tensor = x[1]
		model_prediction = x[2]
		h,w = K.int_shape(model_input)[2:]
		# uniform = K.constant(value=.1)
		# loss = -K.sum(uniform*K.log(model_prediction+K.epsilon()),axis=1,keepdims=True)
		loss = K.categorical_crossentropy(model_prediction, labels_tensor)+0*continuity_loss(model_input,K.int_shape(model_input)[2:])/h*w
		# loss+= K.sum(model_input ** 2, axis=[1, 2, 3], keepdims=False)
		grads = K.gradients(loss, x[0])

		res = K.abs(x[0]-grads[0])
		# loss2= loss+.1*K.sum(K.abs(res-model_input)**2,axis=[1,2,3],keepdims=False)
		# grads2 = K.gradients(loss2,grads[0])
		# grads= grads[0]-grads2[0]
		return grads[0]
		# if self.activation.__name__ == 'relu':
		# 	pas = self.activation(x)
		# 	inv_pas = self.activation(-x)
		# elif self.activation.__name__ == 'sigmoid':
		# 	pas = self.activation(x)
		# 	inv_pas = self.activation(-x)
		# else:
		# 	assert 'activation is not relu'
		# return [pas, inv_pas]

	def get_config(self):
		config = {}
		base_config = super(KnowledgeAvoid, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


def continuity_loss(x, img_shape):
	h, w = img_shape
	assert K.ndim(x) == 4
	a = K.square(x[:, :, :h - 1, :w - 1] - x[:, :, 1:, :w - 1])
	b = K.square(x[:, :, :h - 1, :w - 1] - x[:, :, :h - 1, 1:])
	return K.sum(K.pow(a + b, 1.25))