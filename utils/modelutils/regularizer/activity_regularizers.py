from __future__ import absolute_import
from keras.regularizers import Regularizer
from keras import backend as K
import numpy as np
class VarianceSpatialActivityRegularizer(Regularizer):
	# spatial Variance
	def __init__(self,parameter_dict):
		shape = parameter_dict['shape']
		alpha = parameter_dict['alpha']
		x = np.arange(0,shape[1])
		y = np.arange(0,shape[2])
		z = np.arange(0,shape[0])
		zz,xx,yy = np.meshgrid(z,x,y)
		xx = np.rollaxis(xx, 1)
		yy = np.rollaxis(yy, 1)
		self.mapx = K.cast_to_floatx(xx)
		self.mapy = K.cast_to_floatx(yy)
		self.alpha = K.cast_to_floatx(alpha)
	def __call__(self, x):
		regularization = 0
		regularization+=self.alpha*(K.var(x*self.mapx)+K.var(x*self.mapy))
		return regularization
		# if self.l1:
		# 	regularization += K.sum(self.l1 * K.abs(x))
		# if self.l2:
		# 	regularization += K.sum(self.l2 * K.square(x))
# 		return regularization
#
	def get_config(self):
		return {
			'name': self.__class__.__name__,
			'alpha'  : float(self.alpha),
			}
class NormalizeActivityRegularizer(Regularizer):
	#Regularize the mean of tensor to the average value
	def __init__(self,parameter_dict,weight,avg=0):
		average = parameter_dict['average']
		self.weight = weight
		if avg==0:
			self.average = K.cast_to_floatx(average)
		else:
			self.average = K.cast_to_floatx(avg)
	def __call__(self, x):
		#TODO FIX the NormalizeActivityRegularizer
		regularization_channel= K.mean(K.abs(K.mean(x,1)-self.average))
		reglarization_spatial = K.mean(K.abs(K.mean(x,[2,3])-self.average))
		# regularization += K.mean(K.square(x))
		regularization =self.weight*((regularization_channel**2+reglarization_spatial**2)**.5)
		return regularization

	# if self.l1:
	# 	regularization += K.sum(self.l1 * K.abs(x))
	# if self.l2:
	# 	regularization += K.sum(self.l2 * K.square(x))
	# 		return regularization
	#
	def get_config(self):
		return {
			'name' : self.__class__.__name__,
		}


class OrthogonalActivityRegularizer(Regularizer):
	#TODO implement it later
	# Regularize the mean of tensor to the average value
	def __init__(self, parameter_dict):
		average = parameter_dict['average']
		self.average = K.cast_to_floatx(average)

	def __call__(self, x):
		regularization = 0
		regularization += K.abs((K.mean(x)) - self.average)
		return regularization

	# if self.l1:
	# 	regularization += K.sum(self.l1 * K.abs(x))
	# if self.l2:
	# 	regularization += K.sum(self.l2 * K.square(x))
	# 		return regularization
	#
	def get_config(self):
		return {'name': self.__class__.__name__, }
