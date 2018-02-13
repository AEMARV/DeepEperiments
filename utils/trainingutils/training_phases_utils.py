

from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint,ReduceLROnPlateau,TerminateOnNaN
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.preprocessing import image
from utils.image import ImageDataGenerator
from keras.utils import np_utils

from callbacks.callback_metric_plot import PlotMetrics
from callbacks.callbacks import *
from callbacks.tensorboard.tensor_board import TensorboardVisualizer
from utils import opt_utils
from utils.trainingutils import lr_sched_fun_db

cpu_debug = False


def preprocess_data_phase(opts, data_train, data_test):
    dataset_name = opt_utils.get_dataset_name(opts)
    aug_opts = opt_utils.get_aug_opts(opts)
    file_name_extension_format = '{}_zca:{}_stdn:{}_mn:{}.npy'
    file_name_extension = file_name_extension_format.format(dataset_name, aug_opts['zca_whitening'], aug_opts['featurewise_std_normalization'], aug_opts['featurewise_center'])
    train_name = 'xtrain_' + file_name_extension
    test_name = 'xtest_' + file_name_extension
    train_file_path = '/home/student/Documents/Codes/Python/DeepEperiments/PreprocessData/' + train_name
    test_file_path = '/home/student/Documents/Codes/Python/DeepEperiments/PreprocessData/' + test_name
    if aug_opts['enable']:
        if not os.path.exists(os.path.abspath(train_file_path)):
            print('Using real-time data augmentation.')
            datagen = ImageDataGenerator(featurewise_center=aug_opts['featurewise_center'], featurewise_std_normalization=aug_opts['featurewise_std_normalization'],
                                         zca_whitening=aug_opts['zca_whitening'])  # apply ZCA whitening)
            print('Data Augmentation enabled')
            print(aug_opts)
            datagen.fit(data_train)
            for i in range(data_train.shape[0]):
                data_train[i] = datagen.standardize(data_train[i])
            for i in range(data_test.shape[0]):
                data_test[i] = datagen.standardize(data_test[i])
            np.save(os.path.abspath(train_file_path), data_train)
            np.save(os.path.abspath(test_file_path), data_test)
        else:
            data_train = np.load(os.path.abspath(train_file_path))
            data_test = np.load(os.path.abspath(test_file_path))
    return data_train, data_test


def collect_callbacks(opts):
    callback_list = []
    result_manager = PlotMetrics(opts)
    experiments_abs_path = result_manager.history_holder.dir_abs_path
    callback_list += [result_manager]

    # callback_list += [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)]
    callback_list += [TensorboardVisualizer(log_dir=experiments_abs_path + '/logs', histogram_freq=1, write_graph=True, write_images=False)]
    callback_list += [CSVLogger(filename=experiments_abs_path + '/training.log', separator=',')]
    # callback_list += [EarlyStopping('acc', min_delta=.001, patience=20, mode='max')]
    callback_list += [ModelCheckpoint(result_manager.history_holder.dir_abs_path + '/checkpoint', period=2,save_best_only=True,save_weights_only=True)]
    callback_list += [LearningRateScheduler(lr_sched_fun_db.lr_sched_function_load(opt_utils.get_dataset_name(opts), opt_utils.get_lr_sched_family(opts)))]
    callback_list += [TerminateOnNaN()]
    return callback_list


def data_augmentation_phase(opts):
    datagen = ImageDataGenerator()
    if opts['aug_opts']['enable']:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(  # set input mean to 0 over the dataset
            samplewise_center=opts['aug_opts']['samplewise_center'],  # set each sample mean to 0
            samplewise_std_normalization=opts['aug_opts']['samplewise_std_normalization'],  # divide each input by its std
            rotation_range=opts['aug_opts']['rotation_range'],  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=opts['aug_opts']['width_shift_range'],  # randomly shift images horizontally (fraction of total width)
            height_shift_range=opts['aug_opts']['height_shift_range'],  # randomly shift images vertically (fraction of total height)
            horizontal_flip=opts['aug_opts']['horizontal_flip'],  # randomly flip images
            vertical_flip=opts['aug_opts']['vertical_flip'])  # randomly flip images
        print('Data Augmentation enabled')
        print(opts['aug_opts'])
    return datagen


def load_data(dataset, opts):
    if dataset == 'cifar100':
        (data_train, label_train), (data_test, label_test) = cifar100.load_data()
    elif dataset == 'cifar10':
        (data_train, label_train), (data_test, label_test) = cifar10.load_data()
    else:
        raise ValueError('dataset loader not defined')
    nb_classes = opts['training_opts']['dataset']['nb_classes'];
    label_train = np_utils.to_categorical(label_train, nb_classes)
    label_test = np_utils.to_categorical(label_test, nb_classes)
    data_train = data_train.astype('float32') / 255
    data_test = data_test.astype('float32') / 255
    return (data_train, label_train), (data_test, label_test)


def dataset_stats(generator):
    sum = 0
    mean = 0
    for batch_index in np.arange(0, generator.nb_sample / generator.batch_size):
        sum += np.mean(generator.next()[0], axis=(0, 1, 2))
        if np.log2(batch_index + 1) % 1 == 0:
            sum = sum / (float(batch_index + 1) / 2)
            mean = (sum + mean) / 2
            sum = 0
    return mean


# def train(model, dataset_abs_path, optimizer, opts, cpu_debug=False):
# 	use_costum_training = False
# 	dataset_abs_path = "/home/student/Documents/Git/Datasets/VOC/Keras"
# 	model.compile(loss=opts['optimizer_opts']['loss']['method'], optimizer=optimizer, metrics=['accuracy', 'categorical_accuracy', 'mean_absolute_percentage_error'])
#
# 	imdg = image.ImageDataGenerator()
# 	training_abs_path = os.path.join(dataset_abs_path, "training")
# 	validation_abs_path = os.path.join(dataset_abs_path, "validation")
#
# 	# imdg.mean = np.array([[[116]], [[110]], [[102]]])
#
# 	train_generator = imdg.flow_from_directory(directory=training_abs_path, target_size=model.input_shape[2:], batch_size=opts['training_opts']['batch_size'], seed=0)
#
# 	validation_generator = imdg.flow_from_directory(directory=validation_abs_path, target_size=model.input_shape[2:], batch_size=opts['training_opts']['batch_size'], seed=0)
# 	# mean_training = dataset_stats(train_generator)
#
# 	if cpu_debug:
# 		samples_per_epoch_training = 6
# 		samples_per_epoch_validation = 6
# 	else:
# 		samples_per_epoch_training = train_generator.samples
# 		samples_per_epoch_validation = validation_generator.samples
#
# 	if use_costum_training:
# 		trainer_from_generator(train_generator, opts['training_opts']['batch_size'], model)
# 		validation_from_generator(validation_generator, opts['training_opts']['batch_size'], model)
# 	else:
# 		plotter = PlotMetrics(opts)
# 		experiments_abs_path = plotter.history_holder.dir_abs_path
# 		tensorboard = TensorboardCostum(log_dir=experiments_abs_path + '/logs', histogram_freq=1, write_graph=False, write_images=False)
# 		csv_logger = CSVLogger(filename=experiments_abs_path + '/training.log', separator=',')
# 		lr_sched = LearningRateScheduler(lr_random_multiScale)
# 		early_stopping = EarlyStopping('acc', min_delta=.0001, patience=40, mode='max')
# 		checkpoint = ModelCheckpoint(experiments_abs_path + '/checkpoint', period=10)
# 		callback_list = [plotter, csv_logger, tensorboard, checkpoint]
# 		# if opts['optimizer_opts']['lr'] == -3:
# 		# lr_sched = LearningNIN()
#
# 		callback_list = callback_list + [lr_sched]
# 		model.fit_generator(generator=train_generator, samples_per_epoch=samples_per_epoch_training, nb_epoch=opts['training_opts']['epoch_nb'], validation_data=validation_generator,
# 		                    nb_val_samples=samples_per_epoch_validation, callbacks=callback_list)
