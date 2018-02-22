import keras.backend as K
from keras.regularizers import Regularizer
class Ent_Reg_Softmax(Regularizer):
	"""Regularizer for L1 and L2 regularization.

	# Arguments
		l1: Float; L1 regularization factor.
		l2: Float; L2 regularization factor.
	"""
	def get_config(self):
		return {'softm_coef': float(self.coef)}

	def __init__(self, coef=None,use_link_func=None,link_func=None):
		self.coef = K.cast_to_floatx(coef)
		self.link_func = link_func
		self.use_link_func = use_link_func
	def __call__(self, x):
		if self.use_link_func:
			y = self.link_func(x)
			ent = -y * K.exp(y)
			ent = self.coef * K.sum(ent)
		else:
			xnorm = x - K.logsumexp(x, axis=2, keepdims=True)
			ent = -xnorm * K.exp(xnorm)
			ent = self.coef * K.sum(ent)
		return -ent


class Ent_Reg_Sigmoid(Regularizer):
	def get_config(self):
		return {'sigm_coef': float(self.coef)}

	def __init__(self, coef=None,use_link_func=None,link_func=None):
		self.coef = K.cast_to_floatx(coef)
		self.link_func = link_func
		self.use_link_func = use_link_func
	def __call__(self, x):
		if self.use_link_func:
			y = self.link_func(x)
			y1 = -K.softplus(-y)
			y0 = -K.softplus(y)
			ent1 = -y1 * K.exp(y1)
			ent0 = -y0 * K.exp(y0)
			ent = ent1 + ent0
			ent = self.coef * K.sum(ent)
		else:
			y1 = -K.softplus(-x)
			y0 = -K.softplus(x)
			ent1 = -y1 * K.exp(y1)
			ent0 = -y0 * K.exp(y0)
			ent = ent1 + ent0
			ent = self.coef * K.sum(ent)
		return -ent




