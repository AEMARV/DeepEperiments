import keras.backend as K
import numpy as np
import tensorflow as tf


def tensorboard_weight_visualize(weight):
	w_img = tf.squeeze(weight)
	shape_list = w_img._shape_as_list()
	if len(shape_list) == 1 or len(shape_list) == 2 or len(shape_list) == 3:
		return None
	if len(shape_list) == 0:
		return None
	w_img = tf.reshape(w_img, [shape_list[0] * shape_list[1], shape_list[2], shape_list[3]])
	filters_for_layer = w_img._shape_as_list()[0]
	filter_num_to_show = filters_for_layer
	image_per_row = int(np.floor(np.sqrt(filter_num_to_show)))
	w_img = w_img[:image_per_row ** 2, :, :]
	w_img = w_img - tf.reduce_min(w_img, [1, 2], keep_dims=True)
	w_img = 255 * w_img / (tf.reduce_max(w_img, axis=[1, 2], keep_dims=True) + K.epsilon())
	w_img = tf.reshape(w_img, (image_per_row, image_per_row, shape_list[2], shape_list[3]))

	w_img = tf.reshape(tf.transpose(w_img, [0, 2, 1, 3]), [image_per_row * shape_list[2], image_per_row * shape_list[3], 1])
	w_img = tf.expand_dims(w_img, 0)
	return w_img


def weight_cosine_similarity(weights_raw):
	# weights should have 4 dimensions
	weight_shape = K.int_shape(weights_raw)
	weights_raw = K.transpose(weights_raw)
	vector_weight = K.reshape(weights_raw, (weight_shape[3], -1))
	vector_weight_normalize = K.l2_normalize(vector_weight, 1)
	vector_weight
	weight_covariance = K.abs(K.dot(vector_weight_normalize, K.transpose(vector_weight_normalize)))
	weight_covariance = weight_covariance - K.eye(size=K.int_shape(weight_covariance)[0])
	covariance_image = K.expand_dims(weight_covariance, 0)
	covariance_image = K.expand_dims(covariance_image, 3)
	covariance_max = K.max(weight_covariance, 1)
	dispersion_hist = K.reshape(weight_covariance, (-1,))
	return [dispersion_hist, covariance_max]


def _imgtensor_to_tensorboard_img_compatible(tensor):
	return tf.transpose(tensor, [0, 2, 3, 1])


def get_outbound_tensors_as_list(layer):
	res = []
	for i in range(layer.inbound_nodes.__len__()):
		res += [layer.get_output_at(i)]
	return res


def layer_output_to_imgtensor(tensor, filter_num_to_show_max):
	o_pos = tensor
	shape_o_np = tf.shape(o_pos)
	filters_for_layer = o_pos._shape_as_list()[1]
	filter_num_to_show = np.min((filter_num_to_show_max, filters_for_layer))
	image_per_row = int(np.floor(np.sqrt(filter_num_to_show)))
	filter_num_to_show = image_per_row ** 2

	if not image_per_row ** 2 == filter_num_to_show:
		assert 'Filter_num must be power of two'
	o_pos = tf.slice(o_pos, [0, 0, 0, 0], [shape_o_np[0], filter_num_to_show, shape_o_np[2], shape_o_np[3]])
	o_pos_abs = tf.abs(o_pos)
	o_pos_abs = 255 * o_pos_abs / (tf.reduce_max(o_pos_abs, axis=[2, 3], keep_dims=True) + K.epsilon())

	# o_neg = o_neg - tf.min_reduce(o_neg,axis=[2,3],keep_dims=True)
	o_pos_img = tf.reshape(o_pos_abs, (shape_o_np[0], image_per_row, image_per_row, shape_o_np[2], shape_o_np[3]))
	o_pos_img = tf.reshape(tf.transpose(o_pos_img, [0, 1, 3, 2, 4]), [shape_o_np[0], image_per_row * shape_o_np[2], image_per_row * shape_o_np[3], 1])
	return o_pos_img

# tf.summary.image('{}_out_Positvie'.format(layer.name), o_pos_img)
# tf.summary.image('{}_out_Negative'.format(layer.name), o_neg_img)
