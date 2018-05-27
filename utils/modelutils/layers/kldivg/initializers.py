from keras.initializers import *
from keras.regularizers import *
import numpy as np
import keras.backend as K
import keras as k
from utils.modelutils.layers.kldivg.OPS import *

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
'''Stochastic Parameterizations'''
class Stoch_Param(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False, linker=None, coef=1):
		self.use_link_func = False
		self.linker = linker
		self.coef = coef

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[0])
		nsymbols =  np.float(shape[2])
		lncorners = ndimens * np.log(nsymbols)

		out = np.random.uniform(low=K.epsilon(),high=1- K.epsilon(), size=shape)
		out = -np.log(out)
		out = out/ np.sum(out,axis=2,keepdims=True)
		out = np.log(out)*0
		#if not self.use_link_func:

		return out * self.coef
	def get_log_prob(self, x):
		if self.use_link_func:
			x = self.linker(x)
			y = x/K.sum(x, axis=2, keepdims=True)
			K.log(y)
		else:
			y = x - self.get_log_normalizer(x)
		return y
	def get_prob(self,x):
		if self.use_link_func:
			x = self.linker(x)
			y = x / K.sum(x, axis=2, keepdims=True)
		else:
			y = K.exp(self.get_log_prob(x))

		return y
	def get_normalizer(self,x):
		if self.use_link_func:
			y = K.sum(self.linker(x), axis=2, keepdims=True)
		else:
			y = K.exp(self.get_log_normalizer(x))
		return y
	def get_log_normalizer(self,x):
		if self.use_link_func:
			y = K.log(K.sum(self.linker(x), axis=2, keepdims=True))
		else:
			x = K.permute_dimensions(x,[0,2,1,3])
			y = logsoftstoch(x)
			y = K.permute_dimensions(y,[0,2,1,3])
		return y
	def get_concentration(self, x):
		if self.use_link_func:
			y = self.linker(x)
		else:
			y = K.exp(x)

		return y


class Stoch_Param_Bin(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False,linker=None, coef =1):
		self.use_link_func = False
		self.linker = linker
		self.coef = coef

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1])*np.float(shape[2])*np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.uniform(low=K.epsilon(), high=1 - K.epsilon(), size=shape)
		out = np.log(out)
		return out * self.coef

	def get_log_prob(self, x0,x1):
		if self.use_link_func:
			x0 = self.linker(x0)
			x1 = self.linker(x1)
			y0 = x0 / (x0 + x1)
			y1 = x1 / (x0 + x1)
			y0 = K.log(y0)
			y1 = K.log(y1)
		else:
			L = self.get_log_normalizer(x0,x1)
			y0 = x0 - L
			y1 = x1 - L

		return y0, y1

	def get_prob(self, x0, x1):
		if self.use_link_func:
			x0 = self.linker(x0)
			x1 = self.linker(x1)
			y0 = x0 / (x0 + x1)
			y1 = x1 / (x0 + x1)
		else:
			y0, y1 = self.get_log_prob(x0,x1)
			y0 = K.exp(y0)
			y1 = K.exp(y1)

		return y0, y1

	def get_log_normalizer(self, x0, x1):
		if self.use_link_func:
			y = self.get_normalizer(x0, x1)
			y = K.log(y)
		else:
			xall = K.stack([x0,x1],axis=4)
			y = K.logsumexp(xall,axis=4,keepdims=False)
		return y

	def get_normalizer(self, x0, x1):
		if self.use_link_func:
			y = self.linker(x0) + self.linker(x1)
		else:
			y = K.exp(self.get_log_normalizer(x0, x1))
		return y

	def get_concentration(self, x0, x1):
		if self.use_link_func:
			y0 = self.linker(x0)
			y1 = self.linker(x1)
		else:
			y0 = K.exp(x0)
			y1 = K.exp(x1)

		return y0 , y1


class Stoch_Param_Bias(Initializer):
	def __init__(self, use_link_func=False, linker= None ,coef =1):
		self.use_link_func = use_link_func
		self.linker = K.exp
		self.coef = coef

	def __call__(self, shape, dtype=None):

		out = np.random.uniform(low=K.epsilon(),high=1- K.epsilon(), size=shape)
		out = -np.log(out)
		out = out/ np.sum(out)
		out = np.log(out)
		return out * self.coef

	def get_log_bias(self,x):
		y = self.get_prob_bias(x)
		y = K.clip(y, K.epsilon(), None)
		logprob = K.log(y)
		return logprob

	def get_concentration(self,x):
		y = self.linker(x)
		return y

	def get_prob_bias(self, x):
		y = self.linker(x)
		y = y / self.get_normalizer(x)
		return y

	def get_log_normalizer(self,x):
		normalizer = self.get_normalizer(x)
		return K.log(normalizer)

	def get_normalizer(self,x):
		y = self.linker(x)
		normalizer = K.sum(y)
		return normalizer

'''Log Simplex Parameterizations'''
# Jeffreys Inits
class LogSimplexParSphericalInit(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False, linker=None):
		self.use_link_func = False
		self.linker = linker

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[0])
		nsymbols =  np.float(shape[2])
		lncorners = ndimens * np.log(nsymbols)

		out = np.random.normal(loc=0, scale=1, size=shape)
		out = out / np.sqrt(np.sum(out**2, axis=2, keepdims=True))
		out = out**2
		out = np.log(out)
		#if not self.use_link_func:

		return out
	def get_log_prob(self, x):
		if self.use_link_func:
			x = self.linker(x)
			y = x/K.sum(x, axis=2, keepdims=True)
			K.log(y)
		else:
			y = x - self.get_log_normalizer(x)
		return y
	def get_prob(self,x):
		if self.use_link_func:
			x = self.linker(x)
			y = x / K.sum(x, axis=2, keepdims=True)
		else:
			y = K.exp(self.get_log_prob(x))

		return y
	def get_normalizer(self,x):
		if self.use_link_func:
			y = K.sum(self.linker(x), axis=2, keepdims=True)
		else:
			y = K.exp(self.get_log_normalizer(x))
		return y
	def get_log_normalizer(self,x):
		if self.use_link_func:
			y = K.log(K.sum(self.linker(x), axis=2, keepdims=True))
		else:
			y = K.logsumexp(x, axis=2, keepdims=True)
		return y
	def get_concentration(self, x):
		if self.use_link_func:
			y = self.linker(x)
		else:
			y = K.exp(x)

		return y


class LogSimplexParSphericalInitBin(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False,linker=None):
		self.use_link_func = False
		self.linker = linker

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1])*np.float(shape[2])*np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.normal(loc=0, scale=1, size=shape)
		out = out**2
		out = np.log(out)
		return out

	def get_log_prob(self, x0,x1):
		if self.use_link_func:
			x0 = self.linker(x0)
			x1 = self.linker(x1)
			y0 = x0 / (x0 + x1)
			y1 = x1 / (x0 + x1)
			y0 = K.log(y0)
			y1 = K.log(y1)
		else:
			L = self.get_log_normalizer(x0,x1)
			y0 = x0 - L
			y1 = x1 - L

		return y0, y1

	def get_prob(self, x0, x1):
		if self.use_link_func:
			x0 = self.linker(x0)
			x1 = self.linker(x1)
			y0 = x0 / (x0 + x1)
			y1 = x1 / (x0 + x1)
		else:
			y0, y1 = self.get_log_prob(x0,x1)
			y0 = K.exp(y0)
			y1 = K.exp(y1)

		return y0, y1

	def get_log_normalizer(self, x0, x1):
		if self.use_link_func:
			y = self.get_normalizer(x0, x1)
			y = K.log(y)
		else:
			xall = K.stack([x0,x1],axis=4)
			y = K.logsumexp(xall,axis=4,keepdims=False)
		return y

	def get_normalizer(self, x0, x1):
		if self.use_link_func:
			y = self.linker(x0) + self.linker(x1)
		else:
			y = K.exp(self.get_log_normalizer(x0, x1))
		return y

	def get_concentration(self, x0, x1):
		if self.use_link_func:
			y0 = self.linker(x0)
			y1 = self.linker(x1)
		else:
			y0 = K.exp(x0)
			y1 = K.exp(x1)

		return y0 , y1


class LogSimplexParSphericalInitBias(Initializer):
	def __init__(self, use_link_func=False, linker= None):
		self.use_link_func = use_link_func
		self.linker = K.exp

	def __call__(self, shape, dtype=None):

		out = (np.random.normal(loc=0, scale=1, size=shape)*0.0) + 1
		out = out / np.sqrt(np.sum(out**2))
		out = out **2
		out = np.log(out)
		return out*0

	def get_log_bias(self,x):
		y = self.get_prob_bias(x)
		y = K.clip(y, K.epsilon(), None)
		logprob = K.log(y)
		return logprob

	def get_concentration(self,x):
		y = self.linker(x)
		return y

	def get_prob_bias(self, x):
		y = self.linker(x)
		y = y / self.get_normalizer(x)
		return y

	def get_log_normalizer(self,x):
		normalizer = self.get_normalizer(x)
		return K.log(normalizer)

	def get_normalizer(self,x):
		y = self.linker(x)
		normalizer = K.sum(y)
		return normalizer

# Dirichlet Init
class Dirichlet_Init(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False, linker=None, coef=1):
		self.use_link_func = False
		self.linker = linker
		self.coef = coef

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[0])
		nsymbols =  np.float(shape[2])
		lncorners = ndimens * np.log(nsymbols)

		out = np.random.uniform(low=K.epsilon(),high=1- K.epsilon(), size=shape)
		out = -np.log(out)
		out = out/ np.sum(out,axis=2,keepdims=True)
		out = np.log(out)
		#if not self.use_link_func:

		return out * self.coef
	def get_log_prob(self, x):
		if self.use_link_func:
			x = self.linker(x)
			y = x/K.sum(x, axis=2, keepdims=True)
			K.log(y)
		else:
			y = x - self.get_log_normalizer(x)
		return y
	def get_prob(self,x):
		if self.use_link_func:
			x = self.linker(x)
			y = x / K.sum(x, axis=2, keepdims=True)
		else:
			y = K.exp(self.get_log_prob(x))

		return y
	def get_normalizer(self,x):
		if self.use_link_func:
			y = K.sum(self.linker(x), axis=2, keepdims=True)
		else:
			y = K.exp(self.get_log_normalizer(x))
		return y
	def get_log_normalizer(self,x):
		if self.use_link_func:
			y = K.log(K.sum(self.linker(x), axis=2, keepdims=True))
		else:
			y = K.logsumexp(x, axis=2, keepdims=True)
		return y
	def get_concentration(self, x):
		if self.use_link_func:
			y = self.linker(x)
		else:
			y = K.exp(x)

		return y


class Dirichlet_Init_Bin(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False,linker=None, coef =1):
		self.use_link_func = False
		self.linker = linker
		self.coef = coef

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1])*np.float(shape[2])*np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.uniform(low=K.epsilon(), high=1 - K.epsilon(), size=shape)
		out = np.log(out)
		return out * self.coef

	def get_log_prob(self, x0,x1):
		if self.use_link_func:
			x0 = self.linker(x0)
			x1 = self.linker(x1)
			y0 = x0 / (x0 + x1)
			y1 = x1 / (x0 + x1)
			y0 = K.log(y0)
			y1 = K.log(y1)
		else:
			L = self.get_log_normalizer(x0,x1)
			y0 = x0 - L
			y1 = x1 - L

		return y0, y1

	def get_prob(self, x0, x1):
		if self.use_link_func:
			x0 = self.linker(x0)
			x1 = self.linker(x1)
			y0 = x0 / (x0 + x1)
			y1 = x1 / (x0 + x1)
		else:
			y0, y1 = self.get_log_prob(x0,x1)
			y0 = K.exp(y0)
			y1 = K.exp(y1)

		return y0, y1

	def get_log_normalizer(self, x0, x1):
		if self.use_link_func:
			y = self.get_normalizer(x0, x1)
			y = K.log(y)
		else:
			xall = K.stack([x0,x1],axis=4)
			y = K.logsumexp(xall,axis=4,keepdims=False)
		return y

	def get_normalizer(self, x0, x1):
		if self.use_link_func:
			y = self.linker(x0) + self.linker(x1)
		else:
			y = K.exp(self.get_log_normalizer(x0, x1))
		return y

	def get_concentration(self, x0, x1):
		if self.use_link_func:
			y0 = self.linker(x0)
			y1 = self.linker(x1)
		else:
			y0 = K.exp(x0)
			y1 = K.exp(x1)

		return y0 , y1


class Dirichlet_Init_Bias(Initializer):
	def __init__(self, use_link_func=False, linker= None ,coef =1):
		self.use_link_func = use_link_func
		self.linker = K.exp
		self.coef = coef

	def __call__(self, shape, dtype=None):

		out = np.random.uniform(low=K.epsilon(),high=1- K.epsilon(), size=shape)
		out = -np.log(out)
		out = out/ np.sum(out)
		out = np.log(out)
		return out * self.coef

	def get_log_bias(self,x):
		y = self.get_prob_bias(x)
		y = K.clip(y, K.epsilon(), None)
		logprob = K.log(y)
		return logprob

	def get_concentration(self,x):
		y = self.linker(x)
		return y

	def get_prob_bias(self, x):
		y = self.linker(x)
		y = y / self.get_normalizer(x)
		return y

	def get_log_normalizer(self,x):
		normalizer = self.get_normalizer(x)
		return K.log(normalizer)

	def get_normalizer(self,x):
		y = self.linker(x)
		normalizer = K.sum(y)
		return normalizer

'''Spherical Parameterizations'''
class UnitSphereInit(Initializer):
	def __init__(self, use_link_func=False, linker=K.square, coef=1.0):
		self.use_link_func = use_link_func
		self.linker = K.square
		self.coef = coef
	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[2]) * np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.normal(loc=0, scale=1, size=shape)
		out = out ** self.coef
		out = out / np.sqrt(np.sum(out**2, axis=2, keepdims=True))
		return out

	def get_log_prob(self, x):
		y = self.get_prob(x)
		y = K.clip(y,K.epsilon(),None)
		logprob = K.log(y)
		return logprob

	def get_prob(self, x):
		y = self.linker(x)
		y = y / self.get_normalizer(x)
		return y

	def get_log_normalizer(self, x):
		normalizer = self.get_normalizer(x)
		return K.log(normalizer)

	def get_normalizer(self, x):
		y = self.linker(x)
		normalizer = K.sum(y, axis=2, keepdims=True)
		return normalizer

	def get_concentration(self, x):
		return x**2


class UnitSphereInitBin(Initializer):
	def __init__(self, use_link_func=False, linker=K.square , coef = 1.0):
		self.use_link_func = use_link_func
		self.linker = K.square
		self.coef = coef

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[2]) * np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.normal(loc=0, scale=1, size=shape)
		return out ** self.coef

	def get_log_prob(self,x0,x1):
		y0,y1 = self.get_prob(x0, x1)
		y0 = K.clip(y0, K.epsilon(), None)
		y1 = K.clip(y1, K.epsilon(), None)
		logprob0 = K.log(y0)
		logprob1 = K.log(y1)

		return logprob0, logprob1

	def get_prob(self,x0,x1):
		norm = self.get_normalizer(x0, x1)
		y0 = self.linker(x0)/norm
		y1 = self.linker(x1)/norm
		return y0, y1

	def get_log_normalizer(self,x0,x1):
		norm = self.get_normalizer(x0, x1)
		return K.log(norm)

	def get_normalizer(self,x0,x1):
		norm = self.linker(x0) + self.linker(x1)
		return norm

	def get_concentration(self,x0,x1):
		return x0**2,x1**2


class UnitSphereInitBias(Initializer):
	def __init__(self, use_link_func=False, linker=K.square, coef=1.0):
		self.use_link_func = use_link_func
		self.linker = K.square
		self.coef = coef

	def __call__(self, shape, dtype=None):

		out = (np.random.normal(loc=0, scale=1, size=shape))
		out = (out*0)+1
		out = out / np.sqrt(np.sum(out**2))
		return out

	def get_log_bias(self,x):
		y = self.get_prob_bias(x)
		y = K.clip(y, K.epsilon(), None)
		logprob = K.log(y)
		return logprob

	def get_concentration(self,x):
		y = self.linker(x)
		return y

	def get_prob_bias(self, x):
		y = self.linker(x)
		y = y / self.get_normalizer(x)
		return y

	def get_log_normalizer(self,x):
		normalizer = self.get_normalizer(x)
		return K.log(normalizer)

	def get_normalizer(self,x):
		y = self.linker(x)
		normalizer = K.sum(y)
		return normalizer


class Unit_Sphere_Init_Logit(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out0 = np.random.normal(loc=0, scale=1, size=shape)
		out1 = np.random.normal(loc=0, scale=1, size=shape)
		out = np.log(out0**2) - np.log(out1**2)

		return out
	def get_log_prob(self,diff):
		logprob0 = -K.softplus(diff)
		logprob1 = -K.softplus(-diff)
		return logprob0, logprob1
	def get_prob(self,diff):
		prob0 = K.sigmoid(-diff)
		prob1 = K.sigmoid(diff)
		return prob0,prob1
	def get_log_normalizer(self,diff):

		return diff*0
	def get_normalizer(self,diff):
		return diff*0 + 1


# Exponential init
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


class Exp_Init_Bin(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out_unif = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		out = np.log(out_unif)
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

# Concentration INits
class AlphaInit(Initializer):
	def __init__(self, use_link_func=False,linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker
	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[0])
		nsymbols = np.float(shape[2])
		lncorners = ndimens * np.log(nsymbols)
		uniform_dist = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		#uniform_dist = np.log(uniform_dist)

		#out = uniform_dist / np.sum(uniform_dist, axis=3, keepdims=True)
		out = uniform_dist/np.sum(uniform_dist, axis=2, keepdims=True)
		prop = np.log(np.random.uniform(K.epsilon(), 1 - K.epsilon(), [1, 1, 1, shape[3]]))
		prop = prop / np.sum(prop)
		out = out * prop
		#out = np.log(np.exp(out)-1)
		return out
	def get_regularizer(self):
		return L1L2(0,0)

	def get_concentration(self, alpha0):
		return self.linker(alpha0)

	def get_log_prob(self,alpha):
		p = self.get_prob(alpha)
		p = K.clip(p, K.epsilon(), None)
		logprob = K.log(p)
		return logprob

	def get_prob(self, alpha):
		alpha = self.linker(alpha)
		p = alpha / K.sum(alpha, axis=2, keepdims=True)
		return p

	def get_log_normalizer(self, alpha):
		alpha = self.linker(alpha)
		log_normalizer = K.log(K.sum(alpha, axis=2, keepdims=True))
		return log_normalizer

	def get_normalizer(self, alpha):
		alpha = self.linker(alpha)
		normalizer = K.sum(alpha, axis=2, keepdims=True)
		return normalizer


class AlphaInitBin(Initializer):

	def __init__(self, use_link_func=False,linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[2]) * np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		prop = np.log(np.random.uniform(K.epsilon(),1 - K.epsilon(),[1,1,1,shape[3]]))
		prop = prop/np.sum(prop)
		#out = np.log(out)
		#out = out / np.sum(out, axis=2, keepdims=True)
		#out = out * prop
		#out = np.log(np.exp(out) - 1)
		return out

	def get_regularizer(self):
		return L1L2(0,0)

	def get_concentration(self, alpha0,alpha1):
		return self.linker(alpha0), self.linker(alpha1)

	def get_log_prob(self,alpha0,alpha1):
		p0,p1 = self.get_prob(alpha0,alpha1)
		p0 = K.clip(p0, K.epsilon(), None)
		p1 = K.clip(p1, K.epsilon(), None)
		return K.log(p0),K.log(p1)

	def get_prob(self, alpha0, alpha1):
		z = self.get_normalizer(alpha0, alpha1)
		return alpha0/z , alpha1/z

	def get_log_normalizer(self, alpha0, alpha1):
		log_normalizer = K.log(self.get_normalizer(alpha0, alpha1))
		return log_normalizer

	def get_normalizer(self, alpha0, alpha1):
		alpha0 = self.linker(alpha0)
		alpha1 = self.linker(alpha1)
		z = alpha0 + alpha1
		return z


class AlphaInitBias(Initializer):
	def __init__(self, use_link_func=False,linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker
	def __call__(self, shape, dtype=None):
		uniform_dist = (np.random.uniform(1-K.epsilon(), 1 - K.epsilon(), shape)*0) +1
		out = uniform_dist/np.sum(uniform_dist)
		return out
	def get_regularizer(self):
		return L1L2(0,0)

	def get_concentration(self, alpha0):
		return self.linker(alpha0)

	def get_log_bias(self,alpha):
		p = self.get_prob_bias(alpha)
#		p = K.clip(p,K.epsilon(),None)
		logprob = K.log(p)
		return logprob

	def get_prob_bias(self, alpha):
		alpha = self.linker(alpha)
		p = alpha / K.sum(alpha)
		return p

	def get_log_normalizer(self, alpha):
		alpha = self.get_normalizer(alpha)
		log_normalizer = K.log(alpha)
		return log_normalizer

	def get_normalizer(self, alpha):
		alpha = self.linker(alpha)
		normalizer = K.sum(alpha)
		return normalizer


class LogInit(Initializer):
	def __init__(self, use_link_func=False, linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		uniform_dist = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		uniform_dist = np.log(uniform_dist)
		normal_dist = np.random.normal(0, 0.001, shape)
		return np.log(uniform_dist / np.sum(uniform_dist, axis=2, keepdims=True))

	def get_regularizer(self):
		return None

	def get_concentration(self, alpha0):
		return K.exp(alpha0)

	def get_log_prob(self, alpha):
		return alpha - K.logsumexp(alpha,axis=2,keepdims=True)

	def get_prob(self, alpha):
		logp = self.get_log_prob(alpha)
		p = K.exp(logp)
		return p

	def get_log_normalizer(self, alpha):
		log_normalizer = K.logsumexp(alpha,axis=2,keepdims=True)
		return log_normalizer

	def get_normalizer(self, alpha):
		normalizer = K.exp(self.get_log_normalizer(alpha))
		return normalizer


class LogInitSC(Initializer):
	def __init__(self, use_link_func=False, linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		uniform_dist = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		uniform_dist = np.log(uniform_dist)
		normal_dist = np.random.normal(0, 0.001, shape)
		return np.log(uniform_dist / np.sum(uniform_dist, axis=(0,1,2), keepdims=True))

	def get_regularizer(self):
		return None

	def get_concentration(self, alpha0):
		return K.exp(alpha0)

	def get_log_prob(self, alpha):
		z=  self.get_log_normalizer(alpha)
		return alpha - z

	def get_prob(self, alpha):
		logp = self.get_log_prob(alpha)
		p = K.exp(logp)
		return p

	def get_log_normalizer(self, alpha):
		log_normalizer = K.logsumexp(alpha,axis=(0,1,2),keepdims=True)
		return log_normalizer

	def get_normalizer(self, alpha):
		normalizer = K.exp(self.get_log_normalizer(alpha))
		return normalizer


# Concentrated on a single component distribution
class ConcentratedSC(Initializer):
	def __init__(self, use_link_func=False,linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		uniform_dist = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		#uniform_dist = np.log(uniform_dist)
		uniform_dist = uniform_dist/np.sum(uniform_dist, (0, 1, 2), keepdims=True)
		normal_dist = np.random.normal(0, 1, shape)
		return uniform_dist

	def get_concentration(self, alpha0):
		return self.linker(alpha0)

	def get_log_prob(self, alpha):
		p = self.get_prob(alpha)
		logprob = K.log(p)
		return logprob

	def get_prob(self,alpha):
		conc = self.linker(alpha)
		p = conc/ self.get_normalizer(alpha)
		return p

	def get_log_normalizer(self, alpha):
		norm = self.get_normalizer(alpha)
		log_normalizer = K.log(norm)
		return log_normalizer

	def get_normalizer(self, alpha):
		alpha = self.linker(alpha)
		normalizer = K.sum(alpha, axis=0, keepdims=True)
		normalizer = K.sum(normalizer, axis=1, keepdims=True)
		normalizer = K.sum(normalizer, axis=2, keepdims=True)
		return normalizer


def linker_square(x):
	y = x**2
	return y


def linker_sqrt(x):
	return K.sqrt(x)


def linker_id(x):
	y = K.clip(x, K.epsilon(), None)
	return y


def linker_abs(x):
	y = K.abs(x)
	return y


def linker_softplus(x):
	y = K.softplus(x)
	y = K.clip(y, K.epsilon(), None)
	return y


def linker_exp(x):
	y = K.exp(x)
	return y


def linker_logabs(x):
	y = K.log(1 + K.abs(x))
	return y


def linker_log(x):
	x = K.clip(x,K.epsilon(), 1 - K.epsilon())
	y = -K.log(x)
	return y
