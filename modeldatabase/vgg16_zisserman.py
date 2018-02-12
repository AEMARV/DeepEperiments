# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''


import numpy as np
import warnings
from scipy import io

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D,Dropout
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K, callbacks
from keras.optimizers import SGD
from keras.regularizers import l1,l2
from keras import callbacks
from IPython import embed
from sklearn import metrics

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#
# class mean_norm_batch(callbacks.Callback):
#     def on_batch_begin(self, batch, logs={}):
#         embed()
#         # print(self.)

def VGG16(include_top=True, weights='imagenet',
          input_tensor=None):
    initialization = 'glorot_normal'
    '''Instantiate the VGG16 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1',init=initialization)(
        img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2',init=initialization)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1',init=initialization)(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2',init=initialization)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    #
    # # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1',init=initialization)(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2',init=initialization)(x)
    # x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3',init=initialization)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    #
    # # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1',init=initialization)(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2',init=initialization)(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3',init=initialization)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1',init=initialization)(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2',init=initialization)(x)
    # x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3',init=initialization)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(1024, activation='relu', name='fc1',init=initialization)(x)
        x = Dropout(p=.5)(x)
        x = Dense(256, activation='relu', name='fc2',init=initialization)(x)
        x = Dropout(p=.5)(x)
        x = Dense(20, activation='softmax', name='predictions',init=initialization)(x)

    # Create model
    model = Model(img_input, x)
    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels.h5',
                                        TH_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                        TF_WEIGHTS_PATH,
                                        cache_subdir='models')
            else:
                weights_path = get_file('vgg16_weights_tf_sfdim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

    return model

def categorical_to_normal(y,axis=1):
    return K.argmax(y,axis=axis)
def show_pred_mean(y_true,y_pred):
    y_pred_lable_ind = K.argmax(y_pred,axis=1)
    y_true_lable_ind = K.argmax(y_true,axis=1)

    return K.dot(K.l2_normalize(y_pred_lable_ind,axis=0),K.l2_normalize(y_true_lable_ind,axis=0))
def stats(y_true,y_pred):
    y_true_n = categorical_to_normal(y_true)
    y_pred_n = categorical_to_normal(y_pred)
    return {"max_dif":K.max(K.abs(y_true_n-y_pred_n)),"min_dif":K.min(K.abs(y_true_n-y_pred_n)),"max_abs_val":K.max(
        y_pred_n),"min_abs_val":K.min(y_pred_n)}
def show_pred_mean2(y_true,y_pred):
    return K.mean(y_true)
def show_true_mean(y_true,y_pred):
    return K.mean(y_true)
def false_pos(y_true,y_pred):
    K.shape(y_true)
if __name__ == '__main__':
    # model = VGG16(include_top=True, weights='imagenet')
    model = VGG16(include_top=True, weights=None)
    nb_batch_size = 64
    nb_epoch=4
    #
    #     img_path = 'elephant.jpg'
    #     img = image.load_img(img_path, target_size=(224, 224))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = preprocess_input(x)
    #     print('Input image shape:', x.shape)
    #     model
    #     preds = model.predict(x)
    #     print('Predicted:', decode_predictions(preds))
    # def image_net_gen()
    #     sklea
    #     img_path = '/Imagenet/train/ILSVRC2012_img_train/'
    #     idg = image.ImageDataGenerator();
    sgd = SGD(lr=0.01, momentum=.9, decay=5*1e-4, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy','mean_squared_error'])
    # data_gen_train = image.ImageDataGenerator(rescale=1./255,featurewise_center=True,
    #                                           featurewise_std_normalization=True,samplewise_center=True,
    #                                           samplewise_std_normalization=True)
    data_gen_train = image.ImageDataGenerator(horizontal_flip=False,featurewise_center=True,
                                              featurewise_std_normalization=False)
    # data_gen_train.fit()
    data_gen_train.mean = np.array([123,116,103])
    data_gen_val = image.ImageDataGenerator(featurewise_center=False)
    data_gen_train_flow = data_gen_train.flow_from_directory('../imagenet/train',target_size=(224,224),
                                                             batch_size=nb_batch_size)
    data_gen_val_flow = data_gen_val.flow_from_directory('../imagenet/validation',target_size=(224,224),
                                                         batch_size=nb_batch_size)
    # nb_train_size = data_gen_train_flow.nb_sample
    # nb_batch_val = data_gen_val_flow.nb_sample/nb_batch_size
    # nb_batch_train = data_gen_train_flow.nb_sample/nb_batch_size
    # X = data_gen_train_flow.next()
    # for i_epoch in np.arange(0,nb_epoch):
    #     loss = 0 ;
    #     for i_batch_index in np.arange(0,nb_batch_train):
    #         loss_batch= model.train_on_batch(X[0],X[1])
    #         loss+=loss_batch[1]
    #         print("batch",i_batch_index+1,"/",nb_batch_train,"loss:",loss_batch[0])
    #     loss = loss/i_batch_index
    #     print("epoch",i_epoch," training loss:",loss)
    #     loss = 0
    #     for i_batch_index in np.arange(0,nb_batch_val):
    #         X_val = data_gen_val_flow.next()
    #         loss_batch = model.test_on_batch(X_val[0],X_val[1])
    #         loss+= loss_batch[1]
    #         print("batch",i_batch_index+1,"/",nb_batch_val,"loss:",loss/(i_batch_index+1))
    #     print("epoch:",i_epoch,"validation loss:",loss)


        # a = model.fit_generator(data_gen_train_flow,samples_per_epoch=data_gen_train_flow.nb_sample,nb_epoch=2,
        #                     validation_split=.1)
    a = model.fit_generator(callbacks=[],generator=data_gen_train_flow,
                            samples_per_epoch=data_gen_train_flow.nb_sample,
                            nb_epoch=100,
                        validation_data=data_gen_val_flow,
                        nb_val_samples=data_gen_val_flow.nb_sample
                        )

        # print(a.history)
        # print(model.get_weights())


        # print(dict_rev)
        # print(dict)
        # dict_rev['490']

        # for i in a[-6:]:
        #     # print(dict_rev[i-1])
        # print()

        # data_gen_val_flow = data_gen_val.flow()
        # data_gen_val_flow = data_gen_val.flow('../imagenet3/validation')