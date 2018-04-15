from keras.initializers import *
from keras.regularizers import *
import numpy as np
import keras.backend as K
import keras as k

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

z= 4.0
class Dirichlet_Init(Initializer):
	"""Initializer that generates tensors Uniform on simplex.
	"""

	def __init__(self, use_link_func=False, linker=None):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1]) * np.float(shape[0])
		nsymbols =  np.float(shape[2])
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		out = np.log(out)
		out = np.log(filts) * np.log(out/np.sum(out, axis=2, keepdims=True))
		c = np.log(np.float(shape[3]))/(np.float(shape[0])*np.float(shape[1]))
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
			y = K.logsumexp(x,axis=2,keepdims=True)

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

	def __init__(self, use_link_func=False,linker=None):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		filts = np.float(shape[3])
		ndimens = np.float(shape[1])*np.float(shape[2])*np.float(shape[0])
		nsymbols = 2
		lncorners = ndimens * np.log(nsymbols)
		out = np.random.uniform(K.epsilon(), 1 - K. epsilon(), shape)
		out = np.log(out)
		c = np.log(np.float(shape[3]))/(np.float(shape[0]) * np.float(shape[1]) * np.float(shape[2]))
		out = np.log(filts)*out
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
			y0 = x0 - K.softplus(x1-x0)
			y1 = x1 - K.softplus(x0-x1)

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
			y = x0 + K.softplus(x1-x0)
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

# Unit Sphere -- Jeffreis prior
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


class Unit_Sphere_Init_Bin(Initializer):
	def __init__(self, use_link_func=False):
		self.use_link_func = use_link_func

	def __call__(self, shape, dtype=None):
		out = np.random.normal(loc=0, scale=1, size=shape)
		if not self.use_link_func:
			out = out**2

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
		return y0, y1
	def get_log_normalizer(self,x0,x1):
		norm = x0**2 + x1**2
		return K.log(norm)
	def get_normalizer(self,x0,x1):
		norm = x0**2 + x1**2
		return norm


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
		uniform_dist = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		uniform_dist = np.log(uniform_dist)
		normal_dist = np.random.normal(0, 0.001, shape)
		return uniform_dist/np.sum(uniform_dist, axis=2, keepdims=True)
	def get_regularizer(self):
		return L1L2(0,0)

	def get_concentration(self, alpha0):
		return self.linker(alpha0)

	def get_log_prob(self,alpha):
		p = self.get_prob(alpha)
#		p = K.clip(p,K.epsilon(),None)
		logprob = K.log(p)
		return logprob

	def get_prob(self, alpha):
		alpha = self.linker(alpha)
		p = alpha / K.sum(alpha, axis=2, keepdims=True)
		return p

	def get_log_normalizer(self, alpha):
		alpha = self.linker(alpha)
		log_normalizer = K.log(K.sum(alpha,axis=2,keepdims=True))
		return log_normalizer

	def get_normalizer(self, alpha):
		alpha = self.linker(alpha)
		normalizer = K.sum(alpha,axis=2,keepdims=True)
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
class AlphaInitBin(Initializer):
	def __init__(self, use_link_func=False,linker=K.abs):
		self.use_link_func = use_link_func
		self.linker = linker

	def __call__(self, shape, dtype=None):
		normal_dist = np.random.normal(0,0.001,shape)
		uniform_dist = np.random.uniform(K.epsilon(), 1 - K.epsilon(), shape)
		uniform_dist = uniform_dist/np.sum(uniform_dist,axis=2,keepdims=True)
		return uniform_dist

	def get_concentration(self,alpha0,alpha1):
		return self.linker(alpha0), self.linker(alpha1)

	def get_log_prob(self,alpha0,alpha1):
		alpha0 = self.linker(alpha0)
		alpha1 = self.linker(alpha1)
		p0 = alpha0 /(alpha0 + alpha1)
		p1 = alpha1 /(alpha0 + alpha1)
		return K.log(p0), K.log(p1)

	def get_prob(self, alpha0, alpha1):
		alpha0 = self.linker(alpha0)
		alpha1 = self.linker(alpha1)
		p0 = alpha0 / (alpha0 + alpha1)
		p1 = alpha1 / (alpha0 + alpha1)
		return p0, p1

	def get_log_normalizer(self, alpha0, alpha1):
		alpha0 = self.linker(alpha0)
		alpha1 = self.linker(alpha1)
		normalizer = (alpha0 + alpha1)
		return K.log(normalizer)

	def get_normalizer(self, alpha0, alpha1):
		alpha0 = self.linker(alpha0)
		alpha1 = self.linker(alpha1)
		normalizer = (alpha0 + alpha1)
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
	y=  x**2
	y = K.clip(y, K.epsilon(), None)
	return y
def linker_sqrt(x):
	return K.sqrt(K.abs(x))


def linker_id(x):
	y = K.clip(x, K.epsilon(), None)
	return y


def linker_abs(x):
	y = K.clip(K.abs(x), K.epsilon(), None)
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
