

from keras.callbacks import Callback
from tensorflow.contrib.tensorboard.plugins import projector
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
class TensorBoard(Callback):
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard.
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
        write_images: whether to write model weights to visualize as
            image in Tensorboard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            self.saver = tf.train.Saver()

            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in list(embeddings.keys())}

            config = projector.ProjectorConfig()
            self.embeddings_logs = []

            for layer_name, tensor in list(embeddings.items()):
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                self.embeddings_logs.append(os.path.join(self.log_dir,
                                                         layer_name + '.ckpt'))

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    tensors = self.model.inputs
                feed_dict = dict(list(zip(tensors, val_data)))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        if self.embeddings_freq and self.embeddings_logs:
            if epoch % self.embeddings_freq == 0:
                for log in self.embeddings_logs:
                    self.saver.save(self.sess, log, epoch)

        for name, value in list(logs.items()):
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()
class TensorBoardDefault(Callback):
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard.
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
        write_images: whether to write model weights to visualize as
            image in Tensorboard.
        embeddings_freq: frequency (in epochs) at which selected embedding
            layers will be saved.
        embeddings_layer_names: a list of names of layers to keep eye on. If
            None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name
            in which metadata for this embedding layer is saved. See the
            [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
    """

    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardDefault, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.embeddings_freq = embeddings_freq
        self.embeddings_layer_names = embeddings_layer_names
        self.embeddings_metadata = embeddings_metadata or {}

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq:
            self.saver = tf.train.Saver()

            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']

            embeddings = {layer.name: layer.weights[0]
                          for layer in self.model.layers
                          if layer.name in embeddings_layer_names}

            embeddings_metadata = {}

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in list(embeddings.keys())}

            config = projector.ProjectorConfig()
            self.embeddings_logs = []

            for layer_name, tensor in list(embeddings.items()):
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                self.embeddings_logs.append(os.path.join(self.log_dir,
                                                         layer_name + '.ckpt'))

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    val_data = [val_data[0][0:10, :, :, :], val_data[1]]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    val_data = [val_data[0][0:10, :, :, :], val_data[1], val_data[2]]
                    tensors = self.model.inputs
                feed_dict = dict(list(zip(tensors, val_data)))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

        if self.embeddings_freq and self.embeddings_logs:
            if epoch % self.embeddings_freq == 0:
                for log in self.embeddings_logs:
                    self.saver.save(self.sess, log, epoch)

        for name, value in list(logs.items()):
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

class TensorboardCostum(Callback):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True,layer_list=[],images_num = 4,distribution_sample_size = 128):
        super(TensorboardCostum, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = True

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        filter_num_to_show_max =1024
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    if hasattr(tf, 'histogram_summary'):
                        tf.histogram_summary(weight.name, weight)
                    else:
                        tf.summary.histogram(weight.name, weight)

                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = tf.shape(w_img)
                        shape_list = w_img._shape_as_list()
                        if len(shape_list) == 1 or len(shape_list) == 2 or len(shape_list) == 3:
                            break
                        if len(shape_list)==0:
                            break
                        w_img=tf.reshape(w_img,[shape_list[0]*shape_list[1],shape_list[2],shape_list[3]])
                        filters_for_layer = w_img._shape_as_list()[0]
                        filter_num_to_show = filters_for_layer
                        image_per_row = int(np.floor(np.sqrt(filter_num_to_show)))
                        w_img = w_img[:image_per_row**2,:,:]
                        w_img = w_img - tf.reduce_min(w_img,[1,2],keep_dims=True)
                        w_img = 255 * w_img / (
                        tf.reduce_max(w_img, axis=[1, 2], keep_dims=True) + K.epsilon())
                        w_img = tf.reshape(w_img, (
                        image_per_row, image_per_row, shape_list[2], shape_list[3]))

                        w_img = tf.reshape(tf.transpose(w_img, [ 0, 2, 1, 3]),
                                               [ image_per_row * shape_list[2],
                                                image_per_row * shape_list[3], 1])
                        w_img = tf.expand_dims(w_img,0)



                        if hasattr(tf, 'image_summary'):
                            tf.image_summary(weight.name, w_img)
                        else:
                            tf.summary.image(weight.name, w_img)
                # show image and histogram of Birelus
                if not layer.name.find('image_batch')==-1:
                    o = layer.output
                    o = tf.transpose(o,[0,2,3,1])
                    tf.summary.image('{}_image'.format(layer.name), o)
                if not layer.name.find('CONV')==-1:
                    weights_raw= layer.weights[0]
                    weight_shape = K.int_shape(weights_raw)
                    weights_raw = K.transpose(weights_raw)
                    vector_weight = K.reshape(weights_raw,(weight_shape[3],-1))
                    vector_weight_normalize = K.l2_normalize(vector_weight,1)
                    vector_weight
                    weight_covariance = K.abs(K.dot(vector_weight_normalize,K.transpose(vector_weight_normalize)))
                    weight_covariance = weight_covariance - K.eye(size = K.int_shape(weight_covariance)[0])
                    covariance_image = K.expand_dims(weight_covariance,0)
                    covariance_image = K.expand_dims(covariance_image, 3)
                    covariance_max = K.max(weight_covariance,1)
                    dispersion_hist = K.reshape(weight_covariance,(-1,))
                    tf.summary.histogram(name='{}_Dispersion'.format(layer.name),values=dispersion_hist)
                    tf.summary.histogram(name='{}_Dispersion_Max'.format(layer.name), values=covariance_max)
                    tf.summary.image(name='{}_Dispersion'.format(layer.name),tensor=covariance_image)
                if not layer.name.find('BER')==-1:
                    # out = tf.sl
                    # WEIGHT DISPERSION RATE trying to find how many weights in a layer are simillar

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
                    # tf.summary.image('{}_out_Positvie'.format(layer.name), o_pos_img)
                    # tf.summary.image('{}_out_Negative'.format(layer.name), o_neg_img)
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
                    if not layer.name.find('RELU')==-1 or not layer.name.find('PRelu')==-1:
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
                        o_pos = o_pos+tf.reduce_min(o_pos, axis=[2, 3], keep_dims=True)
                        o_pos = 255 * o_pos / (
                        tf.reduce_max(o_pos, axis=[2, 3], keep_dims=True) + K.epsilon())
                        o_pos = tf.reshape(o_pos,
                                           (shape_o_np[0], image_per_row, image_per_row, shape_o_np[2], shape_o_np[3]))
                        o_pos = tf.reshape(tf.transpose(o_pos, [0, 1, 3, 2, 4]),
                                           [shape_o_np[0], image_per_row * shape_o_np[2], image_per_row * shape_o_np[3],
                                            1])
                        o_pos = tf.concat((tf.zeros_like(o_pos),o_pos,tf.zeros_like(o_pos)),-1)
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

            if self.write_graph:
                if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
                    self.writer = tf.summary.FileWriter(self.log_dir,
                                                        self.sess.graph)
                elif parse_version(tf.__version__) >= parse_version('0.8.0'):
                    self.writer = tf.train.SummaryWriter(self.log_dir,
                                                         self.sess.graph)
                else:
                    self.writer = tf.train.SummaryWriter(self.log_dir,
                                                         self.sess.graph_def)
            if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
                self.writer = tf.summary.FileWriter(self.log_dir)
            else:
                self.writer = tf.train.SummaryWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.histogram_freq and self.validation_data:
            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = self.validation_data[:cut_v_data] + [0]
                    q = [val_data[0][0:10, :, :, :], val_data[1]]
                    val_data =q
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = self.validation_data
                    q = [val_data[0][0:10, :, :, :], val_data[1], val_data[2]]
                    val_data = q
                    tensors = self.model.inputs
                feed_dict = dict(list(zip(tensors, val_data)))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch)

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

class LearningSpringberg(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """

    def __init__(self, S,initial_lr):
        super(LearningSpringberg, self).__init__()
        self.s = S
        self.initial_lr = initial_lr
        self.initial_lr_set = False

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if not self.initial_lr_set:
            K.set_value(self.model.optimizer.lr,self.initial_lr)
            self.initial_lr_set = True
        if epoch in self.s:
            current_lr = K.eval(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, current_lr/10)
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class LearningNIN(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    """
    def __init__(self):
        super(LearningNIN, self).__init__()
        self.s = [0,1,2,3,83,93,103]
        self.initial_lr = [2e-3,1e-2,2e-2,4e-2,4e-3,4e-4]
        self.lr_index =0
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if epoch in self.s:
            K.set_value(self.model.optimizer.lr, self.initial_lr[self.lr_index])
            print('lr_changed to ',str(self.initial_lr[self.lr_index]))
            self.lr_index +=1
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
