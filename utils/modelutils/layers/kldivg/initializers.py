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

class Unit_Sphere_Init(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = K.random_normal(shape=shape,
							  mean=0,
		                      stddev=1,
							  dtype=dtype)

		if not self.use_link_func:
			norm = K.sum(out**2,axis=2,keepdims=True)
			out = out**2/norm
			out = K.log(out)
		return out
	def linkfunc(self,x):
		y = x**2
		y = y/K.sum(y,axis=2,keepdims=True)
		y = K.log(y)
		return y
class Sigmoid_Unit_Sphere_Init(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):

		out0 = K.random_normal(shape=shape,
							  mean=0,
		                      stddev=1,
							  dtype=dtype)



		return out0
	def linkfunc(self,x0,x1):
		norm = x0**2 + x1**2
		y0 = x0**2/norm
		y1 = x1**2/norm
		y0 = K.log(y0)
		y1 = K.log(y1)
		return y0, y1

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
