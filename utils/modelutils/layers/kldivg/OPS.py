import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import keras.backend as K


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
	# Need to generate a unique name to avoid duplicates:
	rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

	tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc": rnd_name}):
		return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


# Def custom square function using np.square instead of tf.square:
def logsoftstoch(x, name=None):
	with ops.name_scope(name, "LogSoftStoch",[x]) as name:
		sqr_x = py_func(_logsumexpf,
		                [x],
		                [tf.float32],
		                name=name,
		                grad=_lsoftstochgrad)  # <-- here's the call to the gradient
		shlist = x.get_shape().as_list()
		shlist[1] =1
		sqr_x[0].set_shape(tf.TensorShape(shlist))
		return sqr_x[0]


# Actual gradient:
def _logsumexpf(x):
	m = np.max(x,axis=1,keepdims=True)
	out = x - m
	y = np.exp(out)
	y = np.log(np.sum(y,axis=1,keepdims=True))
	out = y+m
	return out

def _lsoftstochgrad(op, grad):
	x = op.inputs[0]
	samp_prob = tf.exp(x - tf.reduce_logsumexp(x,axis=1,keepdims=True))
	samples = am_sample(samp_prob)
	selectgrad = grad*samples
	return selectgrad
def am_sample(p):
	cum_prob_1 = tf.cumsum(p, axis=1)
	shape = tf.shape(cum_prob_1)
	randgen = tf.random_uniform(shape,0.0, 1.0)
	randgen = randgen[0:,0:1,0:,0:]
	samp1 = (cum_prob_1 > randgen)
	temp = tf.logical_xor(samp1[0:,1:,0:,0:], samp1[0:,0:-1,0:,0:])#samp1[0:, 1:, 0:, 0:]
	samp1 = tf.concat([samp1[0:,0:1,0:,0:],temp],axis=1)
	samp1 = tf.to_float(samp1)
	return samp1