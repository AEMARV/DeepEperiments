from keras.optimizers import *
import keras.backend as K
if K.backend() == 'tensorflow':
	import tensorflow as tf
class PolarSGD(Optimizer):
	"""Stochastic gradient descent optimizer.

	Includes support for momentum,
	learning rate decay, and Nesterov momentum.

	# Arguments
		lr: float >= 0. Learning rate.
		momentum: float >= 0. Parameter that accelerates SGD
			in the relevant direction and dampens oscillations.
		decay: float >= 0. Learning rate decay over each update.
		nesterov: boolean. Whether to apply Nesterov momentum.
	"""

	def __init__(self, polar_decay=1, lr=0.01, momentum=0., decay=0.,
				 nesterov=False, **kwargs):
		super(PolarSGD, self).__init__(**kwargs)
		with K.name_scope(self.__class__.__name__):
			self.iterations = K.variable(0, dtype='int64', name='iterations')
			self.lr = K.variable(lr, name='lr')
			self.momentum = K.variable(momentum, name='momentum')
			self.decay = K.variable(decay, name='decay')
			self.polar_decay = K.variable(polar_decay,name='polardecay')
		self.initial_decay = decay
		self.nesterov = nesterov

	@interfaces.legacy_get_updates_support
	def get_updates(self, loss, params):
		grads = self.get_gradients(loss, params)
		self.updates = [K.update_add(self.iterations, 1)]

		lr = self.lr
		if self.initial_decay > 0:
			lr *= (1. / (1. + self.decay * K.cast(self.iterations,
												  K.dtype(self.decay))))
		# momentum
		shapes = [K.int_shape(p) for p in params]
		moments = [K.zeros(shape) for shape in shapes]
		self.weights = [self.iterations] + moments
		for p, g, m in zip(params, grads, moments):

			v = self.momentum * m - lr * (g + self.get_weight_decay(p)*self.polar_decay)  # velocity
			self.updates.append(K.update(m, v))

			if self.nesterov:
				new_p = p + self.momentum * v - lr * (g)
			else:
				new_p = p + v

			# Apply constraints.
			if getattr(p, 'constraint', None) is not None:
				new_p = p.constraint(new_p)
			#new_p = new_p - (self.get_weight_decay(new_p)*self.polar_decay * lr)
			self.updates.append(K.update(p, new_p))
		return self.updates
	def get_weight_decay(self,p):
		ndims = K.ndim(p)
		if ndims is 4:
			norm = K.sum(p**2, axis=2, keepdims=True)
		else:
			norm = K.sum(p**2, axis=0, keepdims=True)
		g = p/(K.sqrt(norm))
		#g = p
		return g
	def get_norm_grad(self,g):
		ndims = K.ndim(g)
		if ndims is 4:
			norm = K.sum(g ** 2, axis=2, keepdims=True)
		else:
			norm = K.sum(g ** 2, axis=0, keepdims=True)
		return (norm)
	def get_config(self):
		config = {'lr': float(K.get_value(self.lr)),
				  'momentum': float(K.get_value(self.momentum)),
				  'decay': float(K.get_value(self.decay)),
				  'nesterov': self.nesterov}
		base_config = super(PolarSGD, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

