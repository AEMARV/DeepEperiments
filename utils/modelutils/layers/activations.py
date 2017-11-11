from keras.layers import Layer
import keras.backend as K
import numpy as np
class AmpBER(Layer):
	def __init__(self, **kwargs):
		super(AmpBER, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return [input_shape, input_shape]

	def compute_mask(self, inputs, mask=None):
		return [None, None]

	def call(self, x, mask=None):
		actp_x = K.relu(x)
		actn_x = x - actp_x
		actp_l2 = K.sum(actp_x ** 2, axis=[1, 2, 3], keepdims=True)
		x_l2 = K.sum(x ** 2, axis=[1, 2, 3], keepdims=True)
		actn_l2 = x_l2 - actp_l2
		predp = actp_x * (x_l2 / K.maximum(actp_l2,K.epsilon()))
		predn = actn_x * (x_l2 / K.maximum(actn_l2, K.epsilon()))
		return [predp, predn]


class ReluSplit(Layer):
	def __init__(self,child_nb, **kwargs):
		super(ReluSplit, self).__init__(**kwargs)
		self.child_nb = child_nb

	def compute_output_shape(self, input_shape):
		self.output_filter_nb = int(input_shape[1]/self.child_nb)
		tensor_output_shape = (input_shape[0],self.output_filter_nb,input_shape[2],input_shape[3])

		return self.child_nb*[tensor_output_shape]

	def compute_mask(self, inputs, mask=None):
		return self.child_nb*[None]

	def call(self, x, mask=None):
		pred = K.relu(x)
		output_filter_nb = int(K.int_shape(x)[1]/self.child_nb)
		res = []
		for i in np.arange(self.child_nb):
			res+=[pred[:,output_filter_nb*i:output_filter_nb*(i+1),:,:]]

		return res

class AmpBER1(Layer):
	# selects minimum l2 norm
	def __init__(self, **kwargs):
		super(AmpBER1, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return [input_shape,input_shape]

	def compute_mask(self, inputs, mask=None):
		return [None,None]

	def call(self, x, mask=None):
		actp_x = K.relu(x)
		actn_x = x - actp_x
		actp_l2 = K.sum(actp_x ** 2, axis=[1, 2, 3], keepdims=True)
		x_l2 = K.sum(x ** 2, axis=[1, 2, 3], keepdims=True)
		actn_l2 = x_l2 - actp_l2
		act_l2 = K.stack([actn_l2,actp_l2],axis=4)
		min_indices = K.argmin(act_l2)
		mask = K.one_hot(min_indices,2)
		actn_x = actn_x*mask[:,:,:,:,0]
		actp_x = actn_x * mask[:, :, :, :, 1]
		actn_x = actn_x* x_l2 / (actn_l2+K.epsilon())
		actp_x = actp_x* x_l2 / (actp_l2+K.epsilon())
		return [actn_x,actp_x]


class XLogX(Layer):
	def __init__(self, **kwargs):
		super(XLogX, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		poszero = K.sign(K.sign(x)+1)
		return poszero*K.abs(x+1)*(K.log(K.abs(x+1)+K.epsilon())/(2*np.log(2)))
class AmpRelu(Layer):
	def __init__(self,norm, **kwargs):
		super(AmpRelu, self).__init__(**kwargs)
		self.norm = norm
	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		act_x = K.relu(x)
		act_norm = K.sum(K.abs(act_x) ** self.norm, axis=[1, 2, 3], keepdims=True)
		x_norm = K.sum(K.abs(x) ** self.norm, axis=[1, 2, 3], keepdims=True)
		pred = act_x * (x_norm**(1/self.norm)) / act_norm**(1/self.norm)
		return pred


class AmpBiMeanRelu(Layer):
	def __init__(self, norm, **kwargs):
		super(AmpBiMeanRelu, self).__init__(**kwargs)
		self.norm = norm

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		actp_x = K.relu(x)
		# actn_x = K.relu(-x)
		axis_norm = [1,2,3]
		act_pnorm = K.sum(K.pow(actp_x , self.norm), axis=axis_norm, keepdims=True)
		# act_nnorm = K.sum(K.pow(actn_x ,self.norm), axis=axis_norm, keepdims=True)
		x_norm = K.sum(K.pow(K.abs(x) , self.norm), axis=axis_norm, keepdims=True)
		predp = actp_x * (K.pow(x_norm , (1 / self.norm))) / K.pow(K.maximum(act_pnorm,K.ones_like(act_pnorm)*K.epsilon()) , (1 / self.norm))
		# predn = actn_x * (K.pow(x_norm , (1 / self.norm))) / K.pow((act_nnorm+K.epsilon()) , (1 / self.norm))
		return predp


class LogRelu(Layer):
	def __init__(self, **kwargs):
		super(LogRelu, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		actp_x = K.relu(x)
		actn_x = K.relu(-x)
		actn_x = -K.log(actn_x+1)
		return actp_x+actn_x


class AmpReluch(Layer):
	def __init__(self, **kwargs):
		super(AmpReluch, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		act_x = K.relu(x)
		act_l2 = K.sum(act_x, axis=[1,2, 3], keepdims=True)
		x_l2 = K.sum(K.abs(x) , axis=[1,2, 3], keepdims=True)
		pred = act_x * x_l2 / (act_l2+K.epsilon())
		return pred


class LLU(Layer):
	def __init__(self, **kwargs):
		super(LLU, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape
	def build(self, input_shape):
		self.param = self.add_weight('K',shape=(1,),initializer='ones')
	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		return (self.param*x)+K.log(1+K.exp(-self.param*x))
class AdaRelu(Layer):
	def __init__(self, **kwargs):
		super(AdaRelu, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		self.tensor_list_len = input_shape.__len__
		return input_shape[0]

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		concat_x = K.stack(x, 4)
		pre_act_l2 = K.sum(concat_x ** 2, axis=[1, 2, 3], keepdims=True)
		act_x = K.relu(concat_x)
		act_l2 = K.sum(act_x ** 2, axis=[1, 2, 3], keepdims=True)
		weights = act_l2 / (pre_act_l2 + K.epsilon())
		weights_normalized = weights / K.sum(weights, axis=4, keepdims=True)
		pred = K.sum(weights_normalized * act_x, axis=4)
		return pred

