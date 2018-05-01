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
class MixEntReg(Regularizer):
	"""Regularizer for L1 and L2 regularization.

	# Arguments
		l1: Float; L1 regularization factor.
		l2: Float; L2 regularization factor.
	"""
	def get_config(self):
		return {'softm_coef': float(self.coef)}

	def __init__(self, coef=None,initializer=None):
		self.coef = K.cast_to_floatx(coef)
		self.initializer = initializer
	def __call__(self, x):
		xp = self.initializer.get_prob(x)
		xl = self.initializer.get_log_prob(x)
		avgent = self.avg_entropy(xp,xl)
		mixprob = K.mean(xp,3,keepdims=False)
		mixent = self.avg_entropy(mixprob,K.log(K.clip(mixprob,K.epsilon(),1)))
		deltaent = mixent - avgent
		return -self.coef *deltaent
	def avg_entropy(self,xp,xl):
		h = xp*xl
		h = -K.sum(h,(0,1,2),keepdims=False)
		h = K.mean(h)
		return h

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


class BiasMinEnt(Regularizer):
	def get_config(self):
		return {'BiasMinEnt': float(self.coef)}

	def __init__(self, coef=None,use_link_func=None,link_func=None):
		self.coef = K.cast_to_floatx(coef)
	def __call__(self, x):
		bnorm = x - K.logsumexp(x)
		bexp = K.exp(bnorm)
		bexp = K.clip(bexp,K.epsilon(),1- K.epsilon())
		H = -bexp*bnorm
		H = self.coef * K.sum(H)
		return H
class BiasMaxEnt(Regularizer):
	def get_config(self):
		return {'BiasMaxEnt': float(self.coef)}

	def __init__(self, coef=None,use_link_func=None,link_func=None):
		self.coef = K.cast_to_floatx(coef)
	def __call__(self, x):
		bnorm = x - K.logsumexp(x)
		bexp = K.exp(bnorm)
		bexp = K.clip(bexp, K.epsilon(), 1- K.epsilon())
		H = -bexp*bnorm
		H = self.coef * K.sum(H)
		return -H



