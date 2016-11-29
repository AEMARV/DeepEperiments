import numpy.ma as ma
import numpy as np
import pylab as pl
import matplotlib.cm as cm
from utils import visualizer_utils as vu
class Visualizer():
	# used source : https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
	#FIELDS:
	model = None
	abs_path = ''
	layer_indexes= []
	#END_OF_FIELDS
	def __init__(self,abs_path='',layer_indexes=[]):
		self.abs_path=abs_path
		self.layer_indexes = layer_indexes
	def visualize_layer(self,layer_index,model):
		W = model.layers[layer_index].W.get_value(borrow=True)
		W = np.squeeze(W)
		print("W shape : ", W.shape)
		pl.figure(figsize=(15, 15))
		pl.title(['layer',str(layer_index),'weights'])
		vu.nice_imshow(pl.gca(), vu.make_mosaic(W, np.sqrt(W.shape[0]*W.shape[1])+1, np.sqrt(W.shape[0]*W.shape[1])+1), cmap=cm.binary)



