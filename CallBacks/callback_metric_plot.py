from keras.callbacks import Callback
from matplotlib import pyplot as plt

from ResultManager.history_holder import HistoryHolder


class PlotMetrics(Callback):
	""" Plot the metrics for model using mtaplotlib"""
	TRAIN_LABEL = 'train'
	VALIDATION_LABEL = 'validation'

	def __init__(self):
		plt.ion()
		self.history_holder = HistoryHolder(experiment_name="Testing")
		self.plot_manager = self.history_holder.metric_plot_container
		self.labels = [self.TRAIN_LABEL, self.VALIDATION_LABEL]
		self.metrics = []

	def on_train_begin(self, logs={}):
		for metric_name in self.params['metrics']:
			if metric_name.find('val_') == -1:
				self.metrics += [metric_name]
		self.plot_manager.add_fig_lines_from_list(self.metrics, self.labels)
		self.history_holder.model_plot(self.model)


	def on_epoch_end(self, epoch, logs={}):
		for metric in self.params['metrics']:  # metric is a string eg. val_acc  but metric name is the general metric
			#  name e.g. "acc"
			y = logs.get(metric)
			if metric.find('val_') == -1:
				label = self.TRAIN_LABEL
			else:
				label = self.VALIDATION_LABEL
			name_figure = metric.replace('val_', '')
			self.plot_manager.line_append_point(name_figure, label, x=epoch, y=y)
			# plt.pause(.001)
		self.plot_manager.save_all_fig()
		# visualize layer
		# self.history_holder.weight_visualizer_container.visualize_layer_weights(layer_index_list=[2,4],
		#                                                                         model=self.model,
		#                                                                         filter_channel_index=[(1, 1, 1, 1)
		# 	                                                                        , (1, 2, 3, 4)])
		# plt.pause(.1)
