from __future__ import absolute_import
from __future__ import print_function
from keras.callbacks import Callback

import os
import csv

import numpy as np
import time
import json
import warnings

from collections import deque
from collections import OrderedDict
from collections import Iterable
from keras import backend as K
from pkg_resources import parse_version
import tensorflow as tf
class LearningRateSchedulerCostum(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, schedule,momentum_reset=False):
        super(LearningRateSchedulerCostum, self).__init__()
        self.schedule = schedule
        self.initial_weight =None
        self.momentum_reset = momentum_reset
    def on_train_begin(self, logs=None):
        if self.momentum_reset:
            self.initial_weight = self.model.get_weights()

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch%40 ==0:
            lr =.1**(int(epoch/30)+1)
            destination_weights = self.model.get_weights
            avg = destination_weights-self.initial_weights
            K.set_value(self.model.optimizer.mom)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
class GeneralCallback(Callback):
    def on_train_end(self, logs=None):
        pass
    def on_batch_begin(self, batch, logs=None):
        pass
    def on_batch_end(self, batch, logs=None):
        pass
    def on_epoch_end(self, epoch, logs=None):
        pass
class TensorboardCostum(Callback):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,layer_list=[],images_num = 4,distribution_sample_size = 128):
        super(TensorboardCostum, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        filter_num_to_show_max =16
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                # show image and histogram of Birelus
                if not layer.name.find('image_batch')==-1:
                    o = layer.output
                    o = tf.transpose(o,[0,2,3,1])
                    tf.summary.image('{}_image'.format(layer.name), o)
                if not layer.name.find('E_Birelu_layer')==-1:
                    # out = tf.sl
                    o_pos = layer.output[0]
                    o_neg = layer.output[1]
                    shape_o_np = tf.shape(o_pos)
                    filters_for_layer = o_pos._shape_as_list()[1]
                    filter_num_to_show = np.min((filter_num_to_show_max,filters_for_layer))
                    image_per_row = int(np.floor(np.sqrt(filter_num_to_show)))
                    filter_num_to_show = image_per_row**2
                    if not image_per_row**2==filter_num_to_show:
                        assert 'Filter_num must be power of two'
                    o_pos = tf.slice(o_pos, [0, 0, 0, 0], [shape_o_np[0], filter_num_to_show, shape_o_np[2],
                                                         shape_o_np[3]])
                    o_neg = tf.slice(o_neg, [0, 0, 0, 0],[shape_o_np[0], filter_num_to_show, shape_o_np[2], shape_o_np[
                        3]])
                    o_pos_abs = tf.abs(o_pos)

                    o_pos_abs = 255*o_pos_abs/(tf.reduce_max(o_pos_abs,axis = [2,3],keep_dims=True)+K.epsilon())
                    o_neg_abs = tf.abs(o_neg)
                    # o_neg = o_neg - tf.min_reduce(o_neg,axis=[2,3],keep_dims=True)
                    o_neg_abs = 255 * o_neg_abs / (tf.reduce_max(o_neg_abs, axis=[2, 3], keep_dims=True)+K.epsilon())
                    third_channel = 128*tf.ceil(tf.abs(tf.clip_by_value(o_pos,clip_value_max=0,clip_value_min=-1))) \
                                    +(255*tf.ceil(
                        tf.abs(tf.clip_by_value(o_neg,clip_value_max=0,clip_value_min=-1))))
                    third_channel= tf.reshape(third_channel,
                                       (shape_o_np[0], image_per_row, image_per_row, shape_o_np[2], shape_o_np[3]))
                    third_channel = tf.reshape(tf.transpose(third_channel, [0, 1, 3, 2, 4]),
                                       [shape_o_np[0], image_per_row * shape_o_np[2], image_per_row * shape_o_np[3], 1])
                    o_pos_img = tf.reshape(o_pos_abs,(shape_o_np[0],image_per_row,image_per_row,shape_o_np[2],
                                                      shape_o_np[
                        3]))
                    o_pos_img = tf.reshape(tf.transpose(o_pos_img,[0,1,3,2,4]),[shape_o_np[0],
                                                                               image_per_row*shape_o_np[2],
                                                                      image_per_row*shape_o_np[3],1])
                    o_neg_img = tf.reshape(o_neg_abs,
                                       (shape_o_np[0], image_per_row, image_per_row, shape_o_np[2], shape_o_np[3]))
                    o_neg_img = tf.reshape(tf.transpose(o_neg_img, [0, 1, 3, 2, 4]),
                                       [shape_o_np[0], image_per_row * shape_o_np[2], image_per_row * shape_o_np[3], 1])

                    colord = tf.concat((o_pos_img,o_neg_img,third_channel),-1)
                    tf.summary.image('{}_out_Positvie'.format(layer.name), o_pos_img)
                    tf.summary.image('{}_out_Negative'.format(layer.name), o_neg_img)
                    tf.summary.image('{}_out_Mixed'.format(layer.name), colord)
                    if hasattr(layer, 'output'):
                        if hasattr(tf, 'histogram_summary'):
                            tf.histogram_summary('{}_Positive'.format(layer.name), o_pos)
                        else:
                            tf.summary.histogram('{}_Positive'.format(layer.name), o_pos)
                    if hasattr(layer, 'output'):
                        if hasattr(tf, 'histogram_summary'):
                            tf.histogram_summary('{}_Negative'.format(layer.name), o_neg)
                        else:
                            tf.summary.histogram('{}_Negative'.format(layer.name), o_neg)
                else:
                    if not layer.name.find('R_relu')==-1:
                        o_pos = layer.output
                        shape_o_np = tf.shape(o_pos)
                        filters_for_layer = o_pos._shape_as_list()[1]
                        filter_num_to_show = np.min((filter_num_to_show_max, filters_for_layer))
                        image_per_row = int(np.floor(np.sqrt(filter_num_to_show)))
                        filter_num_to_show = image_per_row ** 2
                        image_per_row = int(np.sqrt(filter_num_to_show))
                        if not image_per_row ** 2 == filter_num_to_show:
                            assert 'Filter_num must be power of two'
                        o_pos = tf.slice(o_pos, [0, 0, 0, 0],
                                         [shape_o_np[0], filter_num_to_show, shape_o_np[2], shape_o_np[3]])
                        o_pos = tf.reshape(o_pos,
                                           (shape_o_np[0], image_per_row, image_per_row, shape_o_np[2], shape_o_np[3]))
                        o_pos = tf.reshape(tf.transpose(o_pos, [0, 1, 3, 2, 4]),
                                           [shape_o_np[0], image_per_row * shape_o_np[2], image_per_row * shape_o_np[3],
                                            1])

                        tf.summary.image('{}_out_Positvie'.format(layer.name), o_pos)
                        if hasattr(layer, 'output'):
                            if hasattr(tf, 'histogram_summary'):
                                tf.histogram_summary('{}_Positive'.format(layer.name), o_pos)
                            else:
                                tf.summary.histogram('{}_Positive'.format(layer.name), o_pos)
        if hasattr(tf, 'merge_all_summaries'):
            self.merged = tf.merge_all_summaries()
        else:
            self.merged = tf.summary.merge_all()
        #
        # if self.write_graph:
        #     if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
        #         self.writer = tf.summary.FileWriter(self.log_dir,
        #                                             self.sess.graph)
        #     elif parse_version(tf.__version__) >= parse_version('0.8.0'):
        #         self.writer = tf.train.SummaryWriter(self.log_dir,
        #                                              self.sess.graph)
        #     else:
        #         self.writer = tf.train.SummaryWriter(self.log_dir,
        #                                              self.sess.graph_def)
        # else:
        if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
            self.writer = tf.summary.FileWriter(self.log_dir)
        else:
            self.writer = tf.train.SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.model.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.model.validation_data[:cut_v_data] + [0]
                    q = [val_data[0][0:10, :, :, :], val_data[1]]
                    val_data =q
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.model.validation_data
                    q = [val_data[0][0:10, :, :, :], val_data[1], val_data[2]]
                    val_data = q
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        K.clear_session()
        self.writer.close()


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if self.monitor.startswith(('acc', 'fmeasure')):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('acc')
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
	        if self.wait >= self.patience:
		        self.stopped_epoch = epoch
		        self.model.stop_training = True
	        self.wait += 1
        if logs.get('acc') == 100:
	        self.stopped_epoch = epoch
	        self.model.stop_training = True
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))
