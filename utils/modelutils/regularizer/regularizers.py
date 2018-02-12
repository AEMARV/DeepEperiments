import keras.backend as K
from keras.activations import tanh
from keras.regularizers import Regularizer
from keras.utils.generic_utils import serialize_keras_object,deserialize_keras_object
import six


def l1_l2_tanh(layer=None,l1=0.01, l2=0.01):
	return L1L2Tanh(l1=l1, l2=l2,layer=layer)
class L1L2Tanh(Regularizer):
	"""Regularizer for L1 and L2 regularization.

	# Arguments
		l1: Float; L1 regularization factor.
		l2: Float; L2 regularization factor.
	"""

	def __init__(self, layer=None, l1=0., l2=0.):
		self.l1 = K.cast_to_floatx(l1)
		self.l2 = K.cast_to_floatx(l2)
		self.layer = layer

	def __call__(self, x):
		regularization = 0.
		if self.l2:
			regularization += K.sum(self.l2 * K.square(self.layer.max_weight*tanh(x*self.layer.slope)))
		return regularization

	def get_config(self):
		return {
			'l1': float(self.l1), 'l2': float(self.l2)
		}


def serialize(regularizer):
	return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
	return deserialize_keras_object(config, module_objects=globals(), custom_objects=custom_objects, printable_module_name='regularizer')


def get(identifier):
	if identifier is None:
		return None
	if isinstance(identifier, dict):
		return deserialize(identifier)
	elif isinstance(identifier, six.string_types):
		config = {'class_name': str(identifier), 'config': {}}
		return deserialize(config)
	elif callable(identifier):
		return identifier
	else:
		raise ValueError('Could not interpret regularizer identifier:', identifier)
