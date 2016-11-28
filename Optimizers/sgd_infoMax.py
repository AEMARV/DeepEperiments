from keras import backend as K
from keras.optimizers import Optimizer
class SGDInfoMax(Optimizer):
	def __init__(self, lr=0.01, momentum=0., decay=0.,
	             nesterov=False, **kwargs):
		super(SGDInfoMax, self).__init__(**kwargs)
		self.__dict__.update(locals())
		self.iterations = K.variable(0.)
		self.lr = K.variable(lr)
		self.momentum = K.variable(momentum)
		self.decay = K.variable(decay)
		self.inital_decay = decay

	def get_updates(self, params, constraints, loss):
		# grads = self.get_gradients(loss, params)
		# grads =
		self.updates = []

		lr = self.lr
		if self.inital_decay > 0:
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
		base_config = super(SGDInfoMax, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

