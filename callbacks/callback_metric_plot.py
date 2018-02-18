from keras.callbacks import Callback
from matplotlib import pyplot as plt

from utils.resultmanager.history_holder import HistoryHolder


class PlotMetrics(Callback):
	""" Plot the metrics for model using mtaplotlib"""
	TRAIN_LABEL = 'train'
	VALIDATION_LABEL = 'validation'

	def __init__(self,opts):
		plt.ion()
		self.history_holder = HistoryHolder(experiment_name=opts['experiment_name'],
		                                    opts=opts,
		                                    )
		# self.plot_manager = self.history_holder.metric_plot_container
		self.labels = [self.TRAIN_LABEL, self.VALIDATION_LABEL]
		self.metrics = []

	def on_train_begin(self, logs={}):
		self.history_holder.model_plot(self.model)
		self.history_holder.store_opts()

