from keras.initializers import *


def compute_fans(shape, data_format='channels_last'):
	"""Computes the number of input and output units for a weight shape.

	# Arguments
		shape: Integer shape tuple.
		data_format: Image data format to use for convolution kernels.
			Note that all kernels in Keras are standardized on the
			`channels_last` ordering (even when inputs are set
			to `channels_first`).

	# Returns
		A tuple of scalars, `(fan_in, fan_out)`.

	# Raises
		ValueError: in case of invalid `data_format` argument.
	"""
	if len(shape) == 2:
		fan_in = shape[0]
		fan_out = shape[1]
	elif len(shape) in {3, 4, 5}:
		# Assuming convolution kernels (1D, 2D or 3D).
		# TH kernel shape: (depth, input_depth, ...)
		# TF kernel shape: (..., input_depth, depth)
		if data_format == 'channels_first':
			receptive_field_size = np.prod(shape[2:])
			fan_in = shape[1] * receptive_field_size
			fan_out = shape[0] * receptive_field_size
		elif data_format == 'channels_last':
			receptive_field_size = np.prod(shape[:2])
			fan_in = shape[-2] * receptive_field_size
			fan_out = shape[-1] * receptive_field_size
		else:
			raise ValueError('Invalid data_format: ' + data_format)
	else:
		# No specific assumptions.
		fan_in = np.sqrt(np.prod(shape))
		fan_out = np.sqrt(np.prod(shape))
	return fan_in, fan_out


class VarianceScalingYingYang(Initializer):
	"""Initializer capable of adapting its scale to the shape of weights.

	With `distribution="normal"`, samples are drawn from a truncated normal
	distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

		- number of input units in the weight tensor, if mode = "fan_in"
		- number of output units, if mode = "fan_out"
		- average of the numbers of input and output units, if mode = "fan_avg"

	With `distribution="uniform"`,
	samples are drawn from a uniform distribution
	within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

	# Arguments
		scale: Scaling factor (positive float).
		mode: One of "fan_in", "fan_out", "fan_avg".
		distribution: Random distribution to use. One of "normal", "uniform".
		seed: A Python integer. Used to seed the random generator.

	# Raises
		ValueError: In case of an invalid value for the "scale", mode" or
		  "distribution" arguments.
	"""

	def __init__(self, scale=1.0, mode='fan_in', distribution='normal', seed=None):
		if scale <= 0.:
			raise ValueError('`scale` must be a positive float. Got:', scale)
		mode = mode.lower()
		if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
			raise ValueError('Invalid `mode` argument: '
			                 'expected on of {"fan_in", "fan_out", "fan_avg"} '
			                 'but got', mode)
		distribution = distribution.lower()
		if distribution not in {'normal', 'uniform'}:
			raise ValueError('Invalid `distribution` argument: '
			                 'expected one of {"normal", "uniform"} '
			                 'but got', distribution)
		self.scale = scale
		self.mode = mode
		self.distribution = distribution
		self.seed = seed

	def __call__(self, shape, dtype=None):
		fan_in, fan_out = compute_fans(shape)
		# since we concat the receptive fields fan in for initialization should be doubled
		fan_in = 2*fan_in
		scale = self.scale
		if self.mode == 'fan_in':
			scale /= max(1., fan_in)
		elif self.mode == 'fan_out':
			scale /= max(1., fan_out)
		else:
			scale /= max(1., float(fan_in + fan_out) / 2)
		if self.distribution == 'normal':
			stddev = np.sqrt(scale)
			return K.truncated_normal(shape, 0., stddev, dtype=dtype, seed=self.seed)
		else:
			limit = np.sqrt(3. * scale)
			return K.random_uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)

	def get_config(self):
		return {
			'scale': self.scale, 'mode': self.mode, 'distribution': self.distribution, 'seed': self.seed
		}