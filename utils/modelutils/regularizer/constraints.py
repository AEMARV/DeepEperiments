from keras.constraints import Constraint
import keras.backend as K
class NonZero(Constraint):
	"""Constrains the weights to be non-negative.
	"""

	def __call__(self, w):
		w *= K.cast(K.greater(w, 0.), K.floatx())
		return w