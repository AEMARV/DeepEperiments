import os

import numpy as np
from keras.models import Model
import matplotlib
from skimage.measure import block_reduce
from skimage.transform import pyramid_gaussian
from skimage.filters.rank import mean
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

WEIGHTS_FILE_RULE = './sci_{}_weights'


def slice_dataset(dataset, labels, sci_number):
	res_dataset = []
	res_labels = []
	total_samples = dataset.shape[0]
	sci_dataset_samples = total_samples / sci_number
	for sci_idx in np.arange(sci_number):
		res_dataset += [dataset[int(sci_idx * sci_dataset_samples):int((sci_idx + 1) * sci_dataset_samples)]]
		res_labels += [labels[int(sci_idx * sci_dataset_samples):int((sci_idx + 1) * sci_dataset_samples)]]
	return res_dataset, res_labels


def save_scientist_model(sci_idx, model: Model):
	model.save_weights(WEIGHTS_FILE_RULE.format(sci_idx))


def load_scientist_model(sci_idx, model: Model):
	if os.path.exists(WEIGHTS_FILE_RULE.format(sci_idx)):
		model.load_weights(WEIGHTS_FILE_RULE.format(sci_idx))


def create_weight_file(model: Model, sci_total):
	for sci_idx in np.arange(sci_total):
		model.save_weights(WEIGHTS_FILE_RULE.format(sci_idx))


# def create_sci_dataset(vanil_dataset_sliced):
# 	res = []
# 	sci_total = len(vanil_dataset_sliced)
# 	for sci_idx in np.arange(sci_total):
def normalize_img(img):
	return (img - np.min(img, axis=(1, 2, 3), keepdims=True)) / (np.max(img, axis=(1, 2, 3), keepdims=True) + 1e-10)


def make_video(images, outimg=None, fps=5, size=None, is_color=True, format="XVID"):
	"""
	Create a video from a list of images.

	@param      outvid      output video
	@param      images      list of images to use in the video
	@param      fps         frame per second
	@param      size        size of each frame
	@param      is_color    color
	@param      format      see http://www.fourcc.org/codecs.php
	@return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

	The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
	By default, the video will have the size of the first image.
	It will resize every image to this size before adding them to the video.
	"""
	from cv2.cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
	fourcc = VideoWriter_fourcc(*format)
	vid = None
	for image in images:
		if not os.path.exists(image):
			raise FileNotFoundError(image)
		img = imread(image)
		if vid is None:
			if size is None:
				size = img.shape[1], img.shape[0]
			vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
		if size[0] != img.shape[1] and size[1] != img.shape[0]:
			img = resize(img, size)
		vid.write(img)
	vid.release()
	return vid


def imshow_compat(img_batch):
	if len(np.shape(img_batch)) == 4:
		im_res = np.transpose(img_batch, [0, 2, 3, 1])
	else:
		im_res = np.transpose(img_batch, [1, 2, 0])
	return im_res
def av_pool(dataset):
	h, w = dataset.shape[2:]
	size_factor = np.random.randint(low = 1,high=int(h/32)+1)
	size = size_factor*32
	x = np.random.randint(0, h - size+1)
	y = np.random.randint(0, w - size+1)
	img_dataset = dataset[:, :, x:x + size, y:y + size]
	block_size = size_factor
	img_dataset = block_reduce(img_dataset,block_size=(1,1,block_size,block_size),func=np.mean)
	return img_dataset, x, y,size
def crop(dataset):
	h,w = dataset.shape[2:]
	x = np.random.randint(0,h-31)
	y = np.random.randint(0,w-31)

	return dataset[:,:,x:x+32,y:y+32],x,y
def transform_dataset(model: Model, vanila_dataset, labels, iterations, epoch, eval_model: Model, optimize=False):
	from cv2.cv2 import VideoWriter, VideoWriter_fourcc,VideoCapture
	fourcc = VideoWriter_fourcc(*'x264')
	res = vanila_dataset
	vids = []
	labels_numeric = np.argmax(labels, axis=1)
	class_find_index = []
	# vids =VideoWriter(filename='a', fourcc=fourcc, fps=float(2),frameSize=(2,3),isColor=True)
	# for i in np.arange(1):
		# FFMpegWriter = manimation.writers['ffmpeg']
		# metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
		# vids+= [FFMpegWriter(fps=15, metadata=metadata)]
		# vids += [VideoWriter(filename='./scimage/noise_signiture_sci_class.avi', fourcc=fourcc, fps=float(2),
		#                      frameSize=vanila_dataset[0].shape[1:], isColor=True)]
		# class_find_index += [np.argmax(labels_numeric == i)]

	# eval_res = eval_model.evaluate(res, labels)
	# print_metrics(eval_res, eval_model.metrics_names)
	lr = 1000 * [10] + 200 * [1] + 300 * [.1] + 400 * [.05]

	for ep in np.arange(iterations):
		print(ep)
		pyramid_gaussian()
		# res_crop,x,y,size = av_pool(res)
		# for class_idx in np.arange(1):
		# 	vids[class_idx].write(imshow_compat(255 * res[1]))
		grads_eval = model.predict([res_crop, labels])
		res_crop = res_crop + ((-1) ** (int(optimize))) * (lr[ep] * grads_eval)
		eval_res = eval_model.evaluate(res_crop[1:30], labels[1:30])
		met = print_metrics(eval_res, eval_model.metrics_names)
		if met == 0:
			break
		# if ep%50==0:
		# 	res = normalize_img(res)
		# res = 1/(1+np.exp(-res))
		res[:,:,x:x+32,y:y+32] =res_crop
		res = np.clip(res, 0, 1)

	# for class_idx in np.arange(1):
	# 	vids[class_idx].release()
	print('max:', np.max(res))

	# eval_res = eval_model.evaluate(res, labels)
	# print_metrics(eval_res, eval_model.metrics_names)
	res = normalize_img(res)
	return res


def print_metrics(results, metrics):
	a = dict(list(zip(metrics, results)))
	for key in a.keys():
		if key in ['acc', 'loss']:
			print('\n', key, '\t', a[key])
	return a['loss']
