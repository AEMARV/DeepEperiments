from keras.initializers import *
import numpy as np
import keras.backend as K
class Sigmoid_Init(Initializer):
	"""Initializer that generates tensors uniform on log odds.
	"""
	def __init__(self,use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = K.random_uniform(shape=shape,
							   minval=K.epsilon(),
							   maxval=1-K.epsilon(),
							   dtype=dtype)
		out = -K.log((1/out)-1)
		return out

	def get_log_prob(self, x):
		log_prob0 = -K.softplus(x)
		log_prob1 = -K.softplus(-x)
		return log_prob0, log_prob1
	def get_prob(selfs,x):
		prob0 = K.sigmoid(-x)
		prob1 = K.sigmoid(x)
		return prob0,prob1
	def get_normalizer(self,x):
		return x*0 +1
	def get_log_normalizer(self,x):
		return x*0

class Dirichlet_Init(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		out = np.log(out)
		if not self.use_link_func:
			out = np.log(out/np.sum(out, axis=2, keepdims=True))
		return out

	def get_log_prob(self, x):
		x = K.abs(x)
		y = x/K.sum(x, axis=2, keepdims=True)
		return K.log(y)
	def get_prob(self,x):
		x = K.abs(x)
		y = x / K.sum(x, axis=2, keepdims=True)
		return y
	def get_normalizer(self,x):
		return K.sum(K.abs(x), axis=2, keepdims=True)
	def get_log_normalizer(self,x):
		return K.log(K.sum(K.abs(x), axis=2, keepdims=True))
class Dirichlet_Init_Bin(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		out = np.log(out)
		if not self.use_link_func:
			raise Exception('Logit Init not using link function')
		return out

	def get_log_prob(self, x0,x1):
		x0 = K.abs(x0)
		x1 = K.abs(x1)
		y0 = x0/(x0+x1)
		y1 = x1/(x0+x1)
		return K.log(y0), K.log(y1)

	def get_prob(self, x0, x1):
		x0 = K.abs(x0)
		x1 = K.abs(x1)
		y0 = x0 / (x0 + x1)
		y1 = x1 / (x0 + x1)
		return y0, y1

	def get_log_normalizer(self,x0,x1):
		return K.log(K.abs(x0)+K.abs(x1))
	def get_normalizer(self,x0,x1):
		return K.abs(x0)+K.abs(x1)


class Unit_Sphere_Init(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = np.random.normal(loc=0, scale=1, size=shape)

		if not self.use_link_func:
			norm = np.sum(out**2, axis=2, keepdims=True)
			out = out**2/norm
			out = np.log(out)
		return out
	def get_log_prob(self,x):
		y = x**2
		normalizer= K.sum(y,axis=2,keepdims=True)
		y = y/normalizer
		logprob = K.log(y)
		return logprob

	def get_prob(self, x):
		y = x ** 2
		normalizer = K.sum(y, axis=2, keepdims=True)
		y = y / normalizer
		return y

	def get_log_normalizer(self,x):
		y = x**2
		normalizer = K.sum(y, axis=2, keepdims=True)
		return K.log(normalizer)
	def get_normalizer(self,x):
		y = x**2
		normalizer = K.sum(y, axis=2, keepdims=True)
		return normalizer
class Unit_Sphere_Init_Logit(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = np.random.normal(loc=0, scale=1, size=shape)
		if not self.use_link_func:
			raise Exception('Logit INIT not using Link Function')

		return out
	def get_log_prob(self,x0,x1):
		norm = x0**2 + x1**2
		y0 = x0**2/norm
		y1 = x1**2/norm
		logprob0 = K.log(y0)
		logprob1 = K.log(y1)
		return logprob0, logprob1
	def get_prob(self,x0,x1):
		norm = x0**2 + x1**2
		y0 = x0**2/norm
		y1 = x1**2/norm
		logprob0 = K.log(y0)
		logprob1 = K.log(y1)
		return logprob0, logprob1
	def get_log_normalizer(self,x0,x1):
		norm = x0**2 + x1**2
		return K.log(norm)
	def get_normalizer(self,x0,x1):
		norm = x0**2 + x1**2
		return norm


class Exp_Init(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out_unif = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		out = np.log(out_unif)
		if not self.use_link_func:
			out = out - K.logsumexp(out, axis=2, keepdims=True)
		return out
	def get_log_prob(self,x):

		y = x - K.logsumexp(x, axis=2, keepdims=True)
		return y
	def get_prob(self,x):

		y = x - K.logsumexp(x, axis=2, keepdims=True)
		return K.exp(y)
	def get_log_normalizer(self,x):
		normalizer = K.logsumexp(x, axis=2, keepdims=True)
		return normalizer
	def get_normalizer(self,x):
		normalizer = K.logsumexp(x, axis=2, keepdims=True)
		return K.exp(normalizer)
class Exp_Init_Logit(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out_unif = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		out = np.log(out_unif)
		if not self.use_link_func:
			out = out - K.logsumexp(out, axis=2, keepdims=True)
		return out
	def get_log_prob(self,x0,x1):

		y1 = -K.softplus(x0-x1)
		y0 = -K.softplus(x1 - x0)
		return y0,y1
	def get_prob(self,x0,x1):
		diff = x0-x1
		y0 = K.sigmoid(diff)
		y1 = K.sigmoid(-diff)
		return y0, y1
	def get_log_normalizer(self,x0,x1):
		normalizer = x0 + K.softplus(x1-x0)
		return normalizer
	def get_normalizer(self,x0,x1):
		normalizer = K.exp(x0) + K.exp(x1)
		return normalizer

