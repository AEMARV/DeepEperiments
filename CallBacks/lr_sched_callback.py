from keras.callbacks import Callback
class LearningRateSchedulerCostum(Callback):
	def __init__(self,scheduler_obj):
		super(LearningRateSchedulerCostum, self).__init__()
		self.scheduler_obj = scheduler_obj
	def on_epoch_begin(self, epoch, logs=None):
		if not hasattr(self.model.optimizer, 'lr'):
			raise ValueError('Optimizer must have a "lr" attribute.')
		lr = self.scheduler_obj