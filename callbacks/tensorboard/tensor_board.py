import keras.backend as K
import tensorflow as tf
from keras.callbacks import Callback
from utils.modelutils.layers.kldivg.layers import *
import numpy as np
from callbacks.tensorboard import tensor_board_utills, tensorboard_layer_viz

HIGH_COMPUTATION = 'high_comp'
LOW_COMPUTATION = 'low_comp'


class TensorboardVisualizer(Callback):
	def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, layer_list=[], images_num=4,
	             distribution_sample_size=128):
		super(TensorboardVisualizer, self).__init__()
		if K.backend() != 'tensorflow':
			raise RuntimeError('TensorBoard callback only works '
			                   'with the TensorFlow backend.')
		global tf
		import tensorflow as tf
		self.log_dir = log_dir
		self.histogram_freq = histogram_freq
		self.merged = None
		self.scalar_summary_merged= None
		self.image_hist_dist_merged= None
		self.image_show_merged = None
		self.write_graph = write_graph
		self.write_images = True
		self.write_grads = True
		self.collections = ['high_comp', 'low_comp']
		self.batch_size =8
		self.val_size  = 64
	def set_model(self, model):
		self.model = model
		self.sess = K.get_session()
		scalar_summary_list =[]
		hist_summary_list=[]
		image_show_list = []
		filter_num_to_show_max = 256
		if self.histogram_freq and self.merged is None:
			for layer in self.model.layers:
				if isinstance(layer, KlConv2DInterface):
					scalar_summary_list += [tf.summary.scalar(name='Entropy_{}'.format(layer.name),
					                                          tensor=layer.avg_entropy())]
					scalar_summary_list += [tf.summary.scalar(name='Bias_Entropy_{}'.format(layer.name),
					                                          tensor=layer.bias_entropy())]
					scalar_summary_list += [tf.summary.scalar(name='Avg_Conc{}'.format(layer.name),
					                                          tensor=layer.avg_concentration())]
				elif isinstance(layer, KlConvBin2DInterface):
					scalar_summary_list += [tf.summary.scalar(name='Entropy_{}'.format(layer.name),
					                                          tensor=layer.avg_entropy())]
					scalar_summary_list += [tf.summary.scalar(name='Bias_Entropy_{}'.format(layer.name),
					                                          tensor=layer.bias_entropy())]
					scalar_summary_list += [tf.summary.scalar(name='Avg_Conc{}'.format(layer.name),
					                                          tensor=layer.avg_concentration())]
				for weight in layer.trainable_weights:

					mapped_weight_name = weight.name.replace(':', '_')
					tf.summary.histogram(mapped_weight_name, weight)
					if self.write_grads:
						grads = model.optimizer.get_gradients(model.total_loss,
						                                      weight)

						def is_indexed_slices(grad):
							return type(grad).__name__ == 'IndexedSlices'

						grads = [
							grad.values if is_indexed_slices(grad) else grad
							for grad in grads]
						tf.summary.histogram('{}_grad'.format(mapped_weight_name), grads)

					# grads = model.optimizer.get_gradients(model.total_loss, weight)
					# hist_summary_list+=[tf.summary.histogram('Gradient_{}_{}'.format(layer.name,weight.name), grads)]
					# hist_summary_list+=[tf.summary.histogram(name='{}_Weight_Dist_{}'.format(layer.name, weight.name), values=weight)]
					#scalar_summary_list += [tf.summary.scalar(name='l2Norm_{}_{}'.format(layer.name, weight.name), tensor=K.mean(weight**2))]
				# if not layer.name.find('image_batch') == -1:
				# 	o = layer.output
				# 	o = tf.transpose(o, [0, 2, 3, 1])
				# 	# hist_summary_list+=[tf.summary.image('{}_image'.format(layer.name), o, max_outputs=1)]
				# 	image_show_list +=[tf.summary.image('{}_image'.format(layer.name), o, max_outputs=1)]
				# elif not layer.name.find('CONV') == -1:
					# tensor_list = tensor_board_utills.get_outbound_tensors_as_list(layer)
					# weights_raw = layer.weights[0]
					# cosine_similarity = tensor_board_utills.weight_cosine_similarity(weights_raw)
					# cosine_similarity_hist = cosine_similarity[0]
					# cosine_similarity_max_hist = cosine_similarity[1]
					# hist_summary_list+=[tf.summary.histogram(name='{}_Dispersion'.format(layer.name), values=cosine_similarity_hist)]
					# hist_summary_list+=[tf.summary.histogram(name='{}_Dispersion_Max'.format(layer.name), values=cosine_similarity_max_hist)]
					# for idx, tensor in enumerate(tensor_list):
					# 	img = tensorboard_layer_viz.multichannel_tensor_image_visualizer(tensor, filter_num_to_show_max)
					# 	image_show_list += [tf.summary.image(name='{}_{}_OUT'.format(layer.name, idx), tensor=img, max_outputs=1)]
				# if not layer.name.find('ACT') == -1:
				# 	image_show_list+=tensorboard_layer_viz.activation_map_image_visualizer(layer, filter_num_to_show_max)
				# elif not layer.name.find('SOFTMAX_Weighted'):
					# for idx, input_tensor in enumerate(layer.input):
					# 	soft_max_input = K.softmax(input_tensor)
					# 	entropy = -K.sum(K.log(soft_max_input + K.epsilon()) * soft_max_input, axis=1)
					# 	average_entropy = K.mean(entropy, axis=0)
					# 	scalar_summary_list+=[tf.summary.scalar(name='SOFTMAX_{}_AVG_ENTROPY'.format(idx), tensor=average_entropy)]
					# 	hist_summary_list += [tf.summary.histogram(name='SOFTMAX_{}_ENTROPYHIST'.format(idx), values=entropy)]
					# pdf = layer.output
					# entropy = -K.sum(K.log(pdf + K.epsilon()) * pdf, axis=1)
					# average_entropy = K.mean(entropy, axis=0)
					# scalar_summary_list += [tf.summary.scalar(name='{}_AVG_ENTROPY'.format(layer.name), tensor=average_entropy)]
					# hist_summary_list += [tf.summary.histogram(name='{}_ENTROPYHIST'.format(layer.name), values=entropy)]
				# elif not layer.name.find('SOFTMAX') == -1:
				# 	tensor_list = tensor_board_utills.get_outbound_tensors_as_list(layer)
				# 	img_tensor = []
				# 	for idx, tensor in enumerate(tensor_list):
				# 		img_tensor +=[tensor]
				# 		pdf = tensor
				# 		entropy = -K.sum(K.log(pdf + K.epsilon()) * pdf, axis=1)
				# 		average_entropy = K.mean(entropy, axis=0)
				# 		scalar_summary_list+=[tf.summary.scalar(name='{}_{}_AVG_ENTROPY'.format(layer.name, idx), tensor=average_entropy)]
				# 		hist_summary_list += [tf.summary.histogram(name='{}_{}_ENTROPYHIST'.format(layer.name, idx), values=entropy)]
				# 	img_tensor = K.stack(img_tensor,axis=2)
				# 	img_tensor = K.expand_dims(img_tensor)
				# 	image_show_list+=[tf.summary.image(name = 'SOFTMAX_BRANCH_PREDICTIONS',tensor=img_tensor,max_outputs=9)]
				# elif layer.outbound_nodes.__len__() == 0:  # TODO not sure the last layer name is out
				# 	pdf = layer.output
				# 	entropy = -K.sum(K.log(pdf + K.epsilon()) * pdf, axis=1)
				# 	average_entropy = K.mean(entropy, axis=0)
				# 	scalar_summary_list += [tf.summary.scalar(name='{}_AVG_ENTROPY'.format(layer.name), tensor=average_entropy)]
				# 	hist_summary_list += [tf.summary.histogram(name='{}_ENTROPYHIST'.format(layer.name), values=entropy)]
			# self.image_hist_dist_merged = tf.summary.merge(hist_summary_list)

			# self.image_show_merged = tf.summary.merge(image_show_list)
			if hasattr(tf, 'merge_all_summaries'):
				self.merged = tf.merge_all_summaries()
			else:
				self.merged = tf.summary.merge_all()

			if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
				self.writer = tf.summary.FileWriter(self.log_dir)
			else:
				self.writer = tf.train.SummaryWriter(self.log_dir)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		# if self.validation_data and self.histogram_freq:
		# 	if epoch % self.histogram_freq == 0:
		#
		val_data = self.validation_data
		tensors = (self.model.inputs + self.model.targets + self.model.sample_weights)

		if self.model.uses_learning_phase:
			tensors += [K.learning_phase()]

		# 		assert len(val_data) == len(tensors)
		# 		val_size = np.minimum(val_data[0].shape[0],self.val_size)
		# 		i = 0
		# 		while i < val_size:
		# 			step = min(self.batch_size, val_size - i)
		# 			batch_val = []
		# 			batch_val.append(val_data[0][i:i + step])
		# 			batch_val.append(val_data[1][i:i + step])
		# 			batch_val.append(val_data[2][i:i + step])
		# 			if self.model.uses_learning_phase:
		# 				batch_val.append(val_data[3])
		# 			feed_dict = dict(list(zip(tensors, batch_val)))
		# 			result = self.sess.run([self.image_hist_dist_merged], feed_dict=feed_dict)
		# 			summary_str = result[0]
		# 			self.writer.add_summary(summary_str, epoch)
		# 			i += self.batch_size
				## Image Show
				# batch_val = []
				# batch_val.append(val_data[0][0:3])
				# batch_val.append(val_data[1][0:3])
				# batch_val.append(val_data[2][0:3])
				# if self.model.uses_learning_phase:
				# 	batch_val.append(val_data[3])
				# feed_dict = dict(list(zip(tensors, batch_val)))
				# result = self.sess.run([self.image_show_merged], feed_dict=feed_dict)
				# summary_str = result[0]
				# self.writer.add_summary(summary_str, epoch)
		"""For scalar values"""
		batch_val=[]
		batch_val.append(val_data[0][0:100])
		batch_val.append(val_data[1][0:100])
		batch_val.append(val_data[2][0:100])
		if self.model.uses_learning_phase:
			batch_val.append(val_data[3])
		result_1 = self.sess.run([self.merged],feed_dict = dict(list(zip(tensors, batch_val))))
		summary_str=result_1[0]
		self.writer.add_summary(summary_str,epoch)
		# if self.histogram_freq and self.validation_data:
		# 	if epoch % self.histogram_freq == 0:
		# 		# TODO: implement batched calls to sess.run
		# 		# (current call will likely go OOM on GPU)
		# 		if self.model.uses_learning_phase:
		# 			cut_v_data = len(self.model.inputs)
		# 			val_data = self.validation_data[:cut_v_data] + [0]
		# 			q = [val_data[0][0:10, :, :, :], val_data[1]]
		# 			val_data = q
		# 			tensors = self.model.inputs + [K.learning_phase()]
		# 		else:
		# 			val_data = self.validation_data
		# 			q = [val_data[0][0:10, :, :, :], val_data[1], val_data[2]]
		# 			val_data = q
		# 			tensors = self.model.inputs
		# 		feed_dict = dict(zip(tensors, val_data))
		# 		result = self.sess.run([self.merged], feed_dict=feed_dict)
		# 		summary_str = result[0]
		# 		self.writer.add_summary(summary_str, epoch)

		for name, value in list(logs.items()):
			if name in ['batch', 'size']:
				continue
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, epoch)
		value = K.eval(self.model.optimizer.lr)
		name = 'lr'
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value.item()
		summary_value.tag = name
		self.writer.add_summary(summary, epoch)
		self.writer.flush()

	def on_train_end(self, _):
		K.clear_session()
		self.writer.close()
