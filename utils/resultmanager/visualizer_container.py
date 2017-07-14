import numpy.ma as ma
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from .figure_container import FigureContainer
from utils import visualizer_utils as vu
class VisualizerContainer(FigureContainer):
	# used source : https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
	VISUALIZER_FIGURE_DEFAULT_PREFIX = 'layer'
	#FIELDS:
	# model = None
	# abs_path = ''
	#END_OF_FIELDS
	def __init__(self,abs_path='',container_id_string=''):
		super(VisualizerContainer,self).__init__(abs_path,container_id_string)
		self.abs_path=abs_path
		self.model = None
	def get_fig_name(self,layer_index):
		return self.VISUALIZER_FIGURE_DEFAULT_PREFIX+str(layer_index)
	def visualize_layer_weights(self,layer_index_list,model,filter_channel_index=None):
		'filter_channel_index : array[(filter_index,channel_index)]'
		for index in layer_index_list:
			if not self.exist_figure(self.get_fig_name(index)):
				self.figure_add(self.get_fig_name(index))
			self.figure_select(name_fig=self.get_fig_name(index))
			self.visualize_layer(layer_index=index,model=model,filter_channel_index =filter_channel_index)
		self.save_all_fig()


	def visualize_layer(self,layer_index,model,filter_channel_index):
		W = model.layers[layer_index].W.get_value(borrow=True)
		W = np.squeeze(W)
		W = W[filter_channel_index]
		plt.title(['layer',str(layer_index),'weights'])
		mosaic =  vu.make_mosaic(W, np.sqrt(W.shape[0]*W.shape[1])+1, int(np.sqrt(W.shape[0]))+1)
		plt.clf()
		vu.nice_imshow(plt.gca(), mosaic,
		               cmap=cm.binary)



