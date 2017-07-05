from keras.regularizers import Regularizer
from keras import backend as K
from keras.activations import softmax
import numpy as np
class SoftmaxEntropyRegularizer(Regularizer):
	# spatial Variance
	def __call__(self, x):
		concat_x = K.stack(x, 2)
		entropy = -K.sum(concat_x * K.log(concat_x), axis=1)
		weights = K.exp(entropy-K.max(entropy,axis=1,keepdims=True))/K.sum(K.exp(entropy-K.max(entropy,axis=1,keepdims=True)),axis=1,keepdims=True)
		weight_entropy = -K.sum(weights*K.log(weights+K.epsilon()),axis=1)
		loss = K.mean(weight_entropy)
		return loss