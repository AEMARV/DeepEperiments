from keras.initializers import *
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

	def linkfunc(self, x):
		return x


class Softmax_Init(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = K.random_uniform(shape=shape,
							   minval=K.epsilon(),
							   maxval=1-K.epsilon(),
							   dtype=dtype)
		out = -K.log(out)
		if not self.use_link_func:
			out = K.log(out/K.sum(out, axis=2, keepdims=True))
		return out

	def linkfunc(self, x):
		x = K.abs(x)
		y = x/K.sum(x, axis=2, keepdims=True)
		y = K.clip(y, K.epsilon(), 1-K.epsilon())
		return K.log(y)


class Exp_Init(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = K.random_uniform(shape=shape,
							   minval=K.epsilon(),
							   maxval=1 - K.epsilon(),
							   dtype=dtype)
		out = -K.log(out)
		return out
	def linkfunc(self,x):
		x = K.abs(x)
		y = -x
		y = y - K.logsumexp(y, axis=2, keepdims=True)
		return -x
