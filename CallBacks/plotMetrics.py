from matplotlib import pyplot as plt
from keras.callbacks import Callback
from ResultManager.plotContainer import PlotContainer
from ResultManager.history_holder import HistoryHolder
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
		self.history_holder = HistoryHolder(experiment_name="Testing")
		self.plot_manager = self.history_holder.plot_container
		self.labels = [self.TRAIN_LABEL,self.VALIDATION_LABEL]
	def on_train_begin(self, logs={}):
		self.metrics =[]
		for metric_name in self.params['metrics']:
			if metric_name.find('val_')==-1:
				self.metrics+=[metric_name]
		self.plot_manager.add_fig_lines_from_list(self.metrics,self.labels)
	def on_epoch_end(self, epoch, logs={}):
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
		print("helllooooooooooooo")
		self.plot_manager.save_all_fig()

		print self.plot_manager.result_dir_abs_path
		print "hellowww222"
		plt.pause(.1)




