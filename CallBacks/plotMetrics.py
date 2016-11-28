from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from keras.callbacks import Callback
from ResultManager.plotContainer import PlotContainer
import numpy as np
class PlotMetrics(Callback):
	""" Plot the metrics for model using mtaplotlib"""
	TRAIN_LABEL = 'train'
	VALIDATION_LABEL = 'validation'
	metrics =None
	history_holder=None
	plot_manager=None
	labels=None
	def __init__(self):
		plt.ion()
		self.plot_manager = PlotContainer()
		self.labels = [self.TRAIN_LABEL,self.VALIDATION_LABEL]
	def on_train_begin(self, logs={}):
		self.metrics =[]
		for metric_name in self.params['metrics']:
			if metric_name.find('val_')==-1:
				self.metrics+=[metric_name]
		self.plot_manager.add_fig_lines_from_list(self.metrics,self.labels)
	def on_epoch_end(self, epoch, logs={}):
		metrics_iterator = logs.iteritems()
		for metric in self.params['metrics']: # metric is a string eg. val_acc  but metric name is the general metric
			#  name e.g. "acc"
			y = logs.get(metric)
			if metric.find('val_')==-1:
				label = self.TRAIN_LABEL
			else:
				label = self.VALIDATION_LABEL
			name_figure = metric.replace('val_','')
			self.plot_manager.line_append_point(name_figure,label,x=epoch,y=y)
			plt.pause(.001)




