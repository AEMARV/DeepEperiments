import keras.backend as K
from keras.optimizers import Optimizer, clip_norm


def kullback_leibler_divergence_uniform(y_true, y_pred):
	y_true = K.clip(y_true, K.epsilon(), 1)
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	uniform = K.constant(.1)
	return K.sum(uniform * K.log(uniform / y_pred), axis=-1)
class SGD(Optimizer):
	"""Stochastic gradient descent optimizer.

	Includes support for momentum,
	learning rate decay, and Nesterov momentum.

	# Arguments
		lr: float >= 0. Learning rate.
		momentum: float >= 0. Parameter updates momentum.
		decay: float >= 0. Learning rate decay over each update.
		nesterov: boolean. Whether to apply Nesterov momentum.
	"""

	def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
		super(SGD, self).__init__(**kwargs)
		self.iterations = K.variable(0., name='iterations')
		self.lr = K.variable(lr, name='lr')
		self.momentum = K.variable(momentum, name='momentum')
		self.decay = K.variable(decay, name='decay')
		self.initial_decay = decay
		self.nesterov = nesterov

	def get_gradients(self, loss, params):

		grads = K.gradients(loss, params)
		if hasattr(self, 'clipnorm') and self.clipnorm > 0:
			norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
			grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
		if hasattr(self, 'clipvalue') and self.clipvalue > 0:
			grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
		return grads

	def get_updates(self, params, constraints, loss):
		yang_params = []
		ying_params = []
		for param in params:
			if 'yang' in param.name:
				yang_params += [param]
			else:
				ying_params+=[param]
		grads = self.get_gradients(loss, ying_params)
		grads_yang = self.get_gradients(kullback_leibler_divergence_uniform,)

		self.updates = []

		lr = self.lr
		if self.initial_decay > 0:
			lr *= (1. / (1. + self.decay * self.iterations))
			self.updates.append(K.update_add(self.iterations, 1))

		# momentum
		shapes = [K.get_variable_shape(p) for p in ying_params]
		moments = [K.zeros(shape) for shape in shapes]
		self.weights = [self.iterations] + moments
		for p, g, m in zip(ying_params, grads, moments):
			v = self.momentum * m - lr * g  # velocity
			self.updates.append(K.update(m, v))

			if self.nesterov:
				new_p = p + self.momentum * v - lr * g
			else:
				new_p = p + v

			# apply constraints
			if p in constraints:
				c = constraints[p]
				new_p = c(new_p)

			self.updates.append(K.update(p, new_p))
		return self.updates

	def get_config(self):
		config = {
			'lr'      : float(K.get_value(self.lr)),
			'momentum': float(K.get_value(self.momentum)),
			'decay'   : float(K.get_value(self.decay)),
			'nesterov': self.nesterov
		}
		base_config = super(SGD, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class SGDYingYang(Optimizer):
	"""Stochastic gradient descent optimizer.

	Includes support for momentum,
	learning rate decay, and Nesterov momentum.

	# Arguments
		lr: float >= 0. Learning rate.
		momentum: float >= 0. Parameter updates momentum.
		decay: float >= 0. Learning rate decay over each update.
		nesterov: boolean. Whether to apply Nesterov momentum.
	"""

	def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
		super(SGDYingYang, self).__init__(**kwargs)
		self.iterations = K.variable(0., name='iterations')
		self.lr = K.variable(lr, name='lr')
		self.momentum = K.variable(momentum, name='momentum')
		self.decay = K.variable(decay, name='decay')
		self.initial_decay = decay
		self.nesterov = nesterov

	def get_gradients(self, loss, params):
		grads = K.gradients(loss, params)
		if hasattr(self, 'clipnorm') and self.clipnorm > 0:
			norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
			grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
		if hasattr(self, 'clipvalue') and self.clipvalue > 0:
			grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
		return grads

	def get_updates(self, params, constraints, loss):
		grads = self.get_gradients(loss, params)
		self.updates = []

		lr = self.lr
		if self.initial_decay > 0:
			lr *= (1. / (1. + self.decay * self.iterations))
			self.updates.append(K.update_add(self.iterations, 1))

		# momentum
		shapes = [K.get_variable_shape(p) for p in params]
		moments = [K.zeros(shape) for shape in shapes]
		self.weights = [self.iterations] + moments
		for p, g, m in zip(params, grads, moments):
			v = self.momentum * m - lr * g  # velocity
			self.updates.append(K.update(m, v))

			if self.nesterov:
				new_p = p + self.momentum * v - lr * g
			else:
				new_p = p + v

			# apply constraints
			if p in constraints:
				c = constraints[p]
				new_p = c(new_p)

			self.updates.append(K.update(p, new_p))
		return self.updates

	def get_config(self):
		config = {
			'lr'      : float(K.get_value(self.lr)),
			'momentum': float(K.get_value(self.momentum)),
			'decay'   : float(K.get_value(self.decay)),
			'nesterov': self.nesterov
		}
		base_config = super(SGDYingYang, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
