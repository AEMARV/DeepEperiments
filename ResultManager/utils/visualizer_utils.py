import numpy.ma as ma
import numpy as np
import pylab as pl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid import make_axes_locatable
def make_mosaic(imgs, nrows, ncols, border=1):
	"""
	Given a set of images with all the same shape, makes a
	mosaic with nrows and ncols
	"""
	if imgs.shape[1]>3:
		nimgs = imgs.shape[0]*imgs.shape[1]
		imshape = imgs.shape[2:]
		imgs = imgs.reshape(imgs.shape[0]*imgs.shape[1],imgs.shape[2],imgs.shape[3])
	else:
		imshape = imgs.shape[1:]
		nimgs = imgs.shape[0]

	mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
	                        ncols * imshape[1] + (ncols - 1) * border),
	                       dtype=np.float32)

	paddedh = imshape[0] + border
	paddedw = imshape[1] + border
	for i in xrange(nimgs):
		row = int(np.floor(i / ncols))
		col = i % ncols
		mosaic[row * paddedh:row * paddedh + imshape[0],
		col * paddedw:col * paddedw + imshape[1]] = imgs[i]
	return mosaic


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
	"""Wrapper around pl.imshow"""
	if cmap is None:
		cmap = cm.jet
	if vmin is None:
		vmin = data.min()
	if vmax is None:
		vmax = data.max()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
	pl.colorbar(im, cax=cax)