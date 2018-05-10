import six
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.utils import serialize_keras_object, deserialize_keras_object
from utils.modelutils.layers.kldivg.layers import *
from modeldatabase.Binary_models.model_constructor_utils import get_model_out_dict, node_list_to_list
from utils import opt_utils
from utils.opt_utils import default_opt_creator
from utils.modelutils.layers.kldivg.initializers import *
from utils.modelutils.layers.kldivg.regularizers import *
from utils.modelutils.layers.kldivg.distances import *
from utils.modelutils.layers.kldivg.kl_models import *
from utils.modelutils.layers.kldivg.models.hellokl import *
def Layer_on_list(layer, tensor_list):
    res = []
    tensor_list = node_list_to_list(tensor_list)
    for x in tensor_list:
        res += [layer(x)]
    return res

# KL Convolutional

def simplenn_BE(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:32,r:5,l2_val:5e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:64,r:5,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:128,r:3,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def simplenn_van(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:32,r:5,l2_val:5e-4->ber' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:64,r:5,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:128,r:3,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:3,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->finvan'
    return get_model_out_dict(opts, model_string=model_string)


def nin_relu_baseline(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5->relu' \
                   '->convsh|f:160,r:1->relu' \
                   '->convsh|f:96,r:1->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5->relu' \
                   '->convsh|f:192,r:1->relu' \
                   '->convsh|f:192,r:1->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3->relu' \
                   '->convsh|f:192,r:1->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh' \
                                                     '->softmax->fin'

    return get_model_out_dict(opts, model_string=model_string)


def nin_relu_baseline_caffe(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1,l2_val:1e-4->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh' \
                                                     '->softmax->fin'

    return get_model_out_dict(opts, model_string=model_string)

def nin_relu_baseline_caffe_lsoft(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->lsoft' \
                   '->convsh|f:160,r:1,l2_val:1e-4->lsoft' \
                   '->convsh|f:96,r:1,l2_val:1e-4->lsoft' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->lsoft' \
                   '->convsh|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->convsh|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->lsoft' \
                   '->convsh|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->convsh|f:' + str(nb_classes) + ',r:1,l2_val:1e-4->lsoft' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh' \
                                                     '->softmax->fin'

    return get_model_out_dict(opts, model_string=model_string)

def nin_besh_caffe(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def nin_besh_caffe2(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def nin_besh_caffe3(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def nin_besh_caffe3(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_debug(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->softmax->max_entropy_branch_select' \
                                                     '->fin'
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_debug2(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->softmax->max_entropy_branch_select' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model1(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->maxaverage_entropy_select|average_rate:.8' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model2(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model3(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'weighted_softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model4(opts, input_shape, nb_classes, getstring_flag=False):
    # last conv is not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->softmax->' \
                   'weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model5(opts, input_shape, nb_classes, getstring_flag=False):
    # all non BER conv is not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->conv|f:160,r:1,l2_val:1e-4->relu' \
                   '->conv|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->conv|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->conv|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->softmax->' \
                   'weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model6(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model7(opts, input_shape, nb_classes, getstring_flag=False):
    # trying activity reg
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:{},r:1->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->softmax_activity_reg->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_perm_nin_model1(opts, input_shape, nb_classes, getstring_flag=False):
    # trying activity reg
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:3,random_permute:0' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:3,random_permute:0' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


""""Structure Test"""


def nin_tree_structure_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->conv|f:{},r:1,l2_val:1e-4->relu' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:{},r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->max_entropy_branch_select' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_6(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->maxav|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_7(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->maxav|r:7,s:1' \
                   '->concat' \
                   '->flattensh' \
                   '->densesh|n:10' \
                   '->softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_8(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_9(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_10(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_11(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_12(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->maxav|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


## EXPERIMENT COLUMNAR


def nin_columnar_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convcolsh|f:24,r:5,l2_val:1e-4,col:8->relucol' \
                   '->convcolsh|f:20,r:3,l2_val:1e-4,col:1->relucol' \
                   '->convcolsh|f:12,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_columnar_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convcolsh|f:24,r:5,l2_val:1e-4,col:4->relucol' \
                   '->convcolsh|f:40,r:3,l2_val:1e-4,col:2->relucol' \
                   '->convcolsh|f:12,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_columnar_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convcolsh|f:48,r:5,l2_val:1e-4,col:4->relucol' \
                   '->convcolsh|f:40,r:3,l2_val:1e-4,col:2->relucol' \
                   '->convcolsh|f:12,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convcolsh|f:48,r:5,l2_val:1e-4,col:4->relucol' \
                   '->convcolsh|f:48,r:1,l2_val:1e-4,col:2->relucol' \
                   '->convcolsh|f:24,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_columnar_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convcolsh|f:24,r:5,l2_val:1e-4,col:8->relucol' \
                   '->convcolsh|f:20,r:3,l2_val:1e-4,col:1->relucol' \
                   '->convcolsh|f:12,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convcolsh|f:24,r:5,l2_val:1e-4,col:8->relucol' \
                   '->convcolsh|f:24,r:1,l2_val:1e-4,col:1->relucol' \
                   '->convcolsh|f:24,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:4' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:2' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:2' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:4' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:4' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:2' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_columnar_ber_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convcolsh|f:96,r:5,l2_val:1e-4,col:2->bercol' \
                   '->convcolsh|f:80,r:3,l2_val:1e-4,col:1->bercol' \
                   '->convcolsh|f:48,r:1,l2_val:1e-4,col:1->bercol' \
                   '->concat_col' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_columnar_ber_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convcolsh|f:48,r:5,l2_val:1e-4,col:4->bercol' \
                   '->convcolsh|f:20,r:3,l2_val:1e-4,col:1->bercol' \
                   '->convcolsh|f:16,r:1,l2_val:1e-4,col:1->bercol' \
                   '->concat_col' \
                   '->ifc|out:2' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convcolsh|f:48,r:5,l2_val:1e-4,col:4->bercol' \
                   '->convcolsh|f:48,r:1,l2_val:1e-4,col:1->relucol' \
                   '->convcolsh|f:48,r:1,l2_val:1e-4,col:1->relucol' \
                   '->concat_col' \
                   '->ifc|out:4' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_6(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_7(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:{},r:1,l2_val:1e-4->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifctanh_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->ifctan|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->ifctan|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:{},r:1,l2_val:1e-4->ber' \
                   '->ifctan|out:1' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_convtanh_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convtansh|f:192,r:5,l2_val:1e-4,k_max:0.2->relu' \
                   '->convtansh|f:160,r:3,l2_val:1e-4,k_max:0.2->relu' \
                   '->convtansh|f:96,r:1,l2_val:1e-4,k_max:0.2->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:5,l2_val:1e-4,k_max:0.2->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.2->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.2->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:3,l2_val:1e-4,k_max:0.2->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.2->relu' \
                   '->convtansh|f:{},r:1,l2_val:1e-4,k_max:0.2->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_convtanh_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convtansh|f:192,r:5,l2_val:1e-4,k_max:1->relu' \
                   '->convtansh|f:160,r:3,l2_val:1e-4,k_max:1->relu' \
                   '->convtansh|f:96,r:1,l2_val:1e-4,k_max:1->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:5,l2_val:1e-4,k_max:1->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:1->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:1->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:3,l2_val:1e-4,k_max:1->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:1->relu' \
                   '->convtansh|f:{},r:1,l2_val:1e-4,k_max:1->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_convtanh_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convtansh|f:192,r:5,l2_val:1e-4,k_max:0.3->relu' \
                   '->convtansh|f:160,r:3,l2_val:1e-4,k_max:0.3->relu' \
                   '->convtansh|f:96,r:1,l2_val:1e-4,k_max:0.3->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:5,l2_val:1e-4,k_max:0.3->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.3->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.3->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:3,l2_val:1e-4,k_max:0.3->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.3->relu' \
                   '->convtansh|f:{},r:1,l2_val:1e-4,k_max:0.3->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_convtanh_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convtansh|f:192,r:5,l2_val:1e-4,k_max:0.5->relu' \
                   '->convtansh|f:160,r:3,l2_val:1e-4,k_max:0.5->relu' \
                   '->convtansh|f:96,r:1,l2_val:1e-4,k_max:0.5->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:5,l2_val:1e-4,k_max:0.5->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.5->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.5->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:3,l2_val:1e-4,k_max:0.5->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-4,k_max:0.5->relu' \
                   '->convtansh|f:{},r:1,l2_val:1e-4,k_max:0.5->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_convtanh_5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convtansh|f:192,r:5,l2_val:1e-6,k_max:1->relu' \
                   '->convtansh|f:160,r:3,l2_val:1e-6,k_max:1->relu' \
                   '->convtansh|f:96,r:1,l2_val:1e-6,k_max:1->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:5,l2_val:1e-6,k_max:1->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-6,k_max:1->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-6,k_max:1->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convtansh|f:192,r:3,l2_val:1e-6,k_max:1->relu' \
                   '->convtansh|f:192,r:1,l2_val:1e-6,k_max:1->relu' \
                   '->convtansh|f:{},r:1,l2_val:1e-6,k_max:1->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_dropneg2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->noise_mul|p:.8,chw:1->relu' \
                   '->convsh|f:160,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2' \
                   '->convsh|f:192,r:5,l2_val:1e-4->noise_mul|p:.5,chw:0->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2' \
                   '->convsh|f:192,r:3,l2_val:1e-4->noise_mul|p:.5,chw:0->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:0' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:0' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm_rand_activity(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->softmax_activity_reg->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm_rand_activity_1(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.6' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->softmax_activity_reg->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm_activity(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:0' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->softmax_activity_reg->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm_rand(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_perm_rand_1(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.6' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_1_perm(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->berp|max_perm:8,random_permute:0' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_1_perm_rand(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->berp|max_perm:8,random_permute:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_1_average(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->ber' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->merge_branch_average' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_structure_2_average(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->merge_branch_average' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


""" PERM EXPERIMENT """


def nin_tree_berp_1(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:3,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:3,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:3,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_berp_2(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_berp_3(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:2,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


# Good results
def nin_tree_biperm_1(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->biperm|max_perm:4,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


# bad Results
def nin_tree_biperm_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->biperm|max_perm:2,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->biperm|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->biperm|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:96,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:{},r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:160,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:96,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:{},r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_5_load(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_perm_6(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.7' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_perm_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_perm_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:8,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:96,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:{},r:1,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_biperm_4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->biperm|max_perm:8,random_permute:1,p:.9' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.9' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_tree_biperm_5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->biperm|max_perm:8,random_permute:1,p:.8' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->berp|max_perm:1,random_permute:1,p:.8' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


"""MAX ENTROPY _ AV ENTROPY _ FULL EXPERIMENT"""


def max_entropy_nin_model8(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model9(opts, input_shape, nb_classes, getstring_flag=False):
    # interconv fc are not shared and is not weighted average
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->merge_branch_average' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model10(opts, input_shape, nb_classes, getstring_flag=False):
    # activity reg
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh' \
                   '->softmax_activity_reg' \
                   '->softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model11(opts, input_shape, nb_classes, getstring_flag=False):
    # same as model 8 but with perm
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:4,random_permute:0' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model12(opts, input_shape, nb_classes, getstring_flag=False):
    # same as model 8 but with random perm
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:4,random_permute:1' \
                   '->conv|f:160,r:1,l2_val:1e-4->relu' \
                   '->conv|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->conv|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->conv|f:192,r:1,l2_val:1e-4->relu' \
                   '->conv|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def max_entropy_nin_model13(opts, input_shape, nb_classes, getstring_flag=False):
    # same as model 8 but with random perm since random perm may work better with shared change every conve to shared
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->berp|max_perm:4,random_permute:1' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax->weighted_average_pred_1' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)


def nin_crelu_caffe2(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->crelu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def nin_crelu_caffe3(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->crelu' \
                   '->convsh|f:160,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->crelu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->crelu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
                                                     '->averagepool|r:7,s:1' \
                                                     '->flattensh->merge_branch_average' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)


def besh_vggcrelu(opts, input_shape, nb_classes, getstring_flag=False):
    nb_filter_list = [32, 32, 64, 128, 64, 64]
    model_string = 'cr|f:64,r:3,b:0,p:1' \
                   '->r|f:64,r:3,b:0,p:1' \
                   '->mp|r:2' \
                   '->cr|f:128,r:3,b:0,p:1' \
                   '->cr|f:128,r:3,b:0,p:1' \
                   '->mp|r:2' \
                   '->rsh|f:256,r:3,b:0,p:1' \
                   '->rsh|f:256,r:3,b:0,p:1' \
                   '->rsh|f:256,r:3,b:0,p:1' \
                   '->mp|r:2' \
                   '->rsh|f:512,r:3,b:0,p:1' \
                   '->rsh|f:512,r:3,b:0,p:1' \
                   '->rsh|f:512,r:3,b:0,p:1' \
                   '->mp|r:2' \
                   '->rsh|f:512,r:3,b:0,p:1' \
                   '->rsh|f:512,r:3,b:0,p:1' \
                   '->rsh|f:512,r:3,b:0,p:1' \
                   '->mp|r:2' \
                   '->flattensh|null:1' \
                   '->densesh|n:4096,act:relu' \
                   '->dropoutsh|p:.5' \
                   '->densesh|n:4096,act:relu' \
                   '->dropoutsh|p:.5' \
                   '->densesh|n:-1' \
                   '->merge|mode:sum'

    if getstring_flag:
        return {'string': model_string}
    model = get_model_out_dict(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list)
    return model


def besh_vgg_baseline(opts, input_shape, nb_classes, getstring_flag=False):
    model = VGG16(False, input_shape=input_shape)
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    return model


# YING YANG

# def nin_baseline(opts, input_shape, nb_classes, getstring_flag=False):
# 	model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
# 	               '->convsh|f:160,r:3,l2_val:1e-4->relu' \
# 	               '->convsh|f:96,r:1,l2_val:1e-4->relu' \
# 	               '->maxpool|r:3,s:2->dropout|p:.5' \
# 	               '->convsh|f:192,r:5,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->averagepool|r:3,s:2->dropout|p:.5' \
# 	               '->convsh|f:192,r:3,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->convsh|f:{},r:1,l2_val:1e-4->relu' \
# 	               '->averagepool|r:7,s:1' \
# 	               '->flattensh->' \
# 	               'softmax' \
# 	               '->fin'.format(nb_classes)
#
# 	return get_model_out_dict(opts, model_string=model_string)
#
#
# def nin_baseline2(opts, input_shape, nb_classes, getstring_flag=False):
# 	model_string = 'convsh|f:192,r:5,l2_val:1e-4->bnsh->relu' \
# 	               '->convsh|f:160,r:3,l2_val:1e-4->relu' \
# 	               '->convsh|f:96,r:1,l2_val:1e-4->relu' \
# 	               '->maxpool|r:3,s:2->dropout|p:.5' \
# 	               '->convsh|f:192,r:5,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->averagepool|r:3,s:2->dropout|p:.5' \
# 	               '->convsh|f:192,r:3,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
# 	               '->averagepool|r:7,s:1' \
# 					'->convsh|f:{},r:1,l2_val:1e-4->relu'\
# 	               '->flattensh->' \
# 	               'softmax' \
# 	               '->fin'.format(nb_classes)
#
# 	return get_model_out_dict(opts, model_string=model_string)



    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline_average_rand(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'random_average->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->finvan'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

def nin_baselinefinvan(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->finvan'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

# BATCHNORM
def nin_baseline_bnsh(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bnsh_ger(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->ger|f:160,r:1,l2_val:5e-4,bias:0' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->ger|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bnsh_ger1c(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:96,r:5,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:3,l2_val:5e-4,bias:0' \
                   '->convsh|f:80,r:3,l2_val:5e-4,bias:0->bnsh->ger1c|f:80,r:1,l2_val:5e-4,bias:0' \
                   '->convsh|f:48,r:1,l2_val:5e-4,bias:0->bnsh->ger1c|f:48,r:1,l2_val:5e-4,bias:0' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:3,l2_val:5e-4,bias:0' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:96,r:3,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:3,l2_val:5e-4,bias:0' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger1c|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bnsh_ger2c(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:96,r:5,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:3,l2_val:5e-4,bias:1' \
                   '->convsh|f:80,r:3,l2_val:5e-4,bias:0->bnsh->ger2c|f:80,r:1,l2_val:5e-4,bias:1' \
                   '->convsh|f:48,r:1,l2_val:5e-4,bias:0->bnsh->ger2c|f:48,r:1,l2_val:5e-4,bias:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:3,l2_val:5e-4,bias:1' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:1,l2_val:5e-4,bias:1' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:1,l2_val:5e-4,bias:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:96,r:3,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:3,l2_val:5e-4,bias:1' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:1,l2_val:5e-4,bias:1' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ger2c|f:96,r:1,l2_val:5e-4,bias:1' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bnsh_gertree(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'gertree|f:3,r:3,l2_val:5e-4,bias:1' \
                   '->gertree|f:3,r:3,l2_val:5e-4,bias:1' \
                   '->gertree|f:3,r:3,l2_val:5e-4,bias:1' \
                   '->gertree|f:3,r:3,l2_val:5e-4,bias:1' \
                   '->gertree|f:3,r:3,l2_val:5e-4,bias:1' \
                   '->gertree|f:3,r:3,l2_val:5e-4,bias:1' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bnsh_gertree_binary_full(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'gertreebinary|f:16,r:5,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:3,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:1,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->gertreebinary|f:16,r:5,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:1,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:1,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->gertreebinary|f:16,r:3,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:1,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:1,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline_bnsh_gertree_binary(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'gertreebinary|f:16,r:5,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:5,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:5,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->gertreebinary|f:16,r:5,l2_val:5e-4,bias:1,mactivation:sigmoid' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline_xlogx(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->xlogx' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_llu(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->llu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline2_ifc_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_ns(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:80,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_sns(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsns|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsns|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsns|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsns|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

def nin_baseline2_ifc_bn_sns2v2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsns|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsns|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsns|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsns|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_snsrecv2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrec|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrec|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
# CRV Experiments
def vgg_baseline(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.3' \
                   '->convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:128,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f:128,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:256,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f:256,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f:256,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f: 512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                   '->convsh|f: 512,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2->dropout|p:.5' \
                   '->convsh|f: 512,r: 1,l2_val:5e-4,bias:1->relu->bn->dropout|p:.5' \
                   '->convshfixedfilter|f:{},r:1,bias:0' \
                   '->flattensh->softmax->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def vgg_ifc_bn_snsrecv2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.3' \
                   '->convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:128,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f:128,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                   '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                   '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                   '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                   '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f: 512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                   '->convsh|f: 512,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2->dropout|p:.5' \
                    '->convsh|f: 512,r: 1,l2_val:5e-4,bias:1->relu->bn->dropout|p:.5' \
                    '->convshfixedfilter|f:{},r:1,bias:0'\
                   '->flattensh->softmax->fin'.format(nb_classes)


    return get_model_out_dict(opts, model_string=model_string)


def vgg_ifc_bn_snsrecv2_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.3' \
                   '->convsnsrec|f:64,r:3,l2_val:5e-4,bias:1->ber->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:128,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsnsrec|f:128,r: 3,l2_val:5e-4,bias:1->ber->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                   '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                   '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                   '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                   '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f: 512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                   '->convsh|f: 512,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2->dropout|p:.5' \
                   '->convsh|f: 512,r: 1,l2_val:5e-4,bias:1->relu->bn->dropout|p:.5' \
                   '->convshfixedfilter|f:{},r:1,bias:0' \
                   '->flattensh->softmax->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def vgg_ifc_bn_snsrecv2_3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.3' \
                   '->convsnsrec|f:64,r:3,l2_val:5e-4,bias:1->ber->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:128,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsnsrec|f:128,r: 3,l2_val:5e-4,bias:1->ber->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                   '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                   '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                   '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                   '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->relu->bn' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:2,s:2' \
                   '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                   '->convsh|f: 512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                   '->convsh|f: 512,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                   '->maxpool|r:2,s:2->dropout|p:.5' \
                   '->convsh|f: 512,r: 1,l2_val:5e-4,bias:1->relu->bn->dropout|p:.5' \
                   '->convshfixedfilter|f:{},r:1,bias:0' \
                   '->flattensh->softmax->fin'.format(nb_classes)

    def vgg_ifc_bn_classic(opts, input_shape, nb_classes, getstring_flag=False):
        model_string = 'convsh|f:64,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.3' \
                       '->convsnsrec|f:64,r:3,l2_val:5e-4,bias:1->ber->bn' \
                       '->ifcv2|out:1' \
                       '->maxpool|r:2,s:2' \
                       '->convsh|f:128,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                       '->convsnsrec|f:128,r: 3,l2_val:5e-4,bias:1->ber->bn' \
                       '->ifcv2|out:1' \
                       '->maxpool|r:2,s:2' \
                       '->convsh|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                       '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p:.4' \
                       '->convsnsrec|f:256,r:3,l2_val:5e-4,bias:1->relu->bn' \
                       '->ifcv2|out:1' \
                       '->maxpool|r:2,s:2' \
                       '->convsh|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                       '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->ber->bn->dropout|p: .4' \
                       '->convsnsrec|f:512,r:3,l2_val:5e-4,bias:1->relu->bn' \
                       '->ifcv2|out:1' \
                       '->maxpool|r:2,s:2' \
                       '->convsh|f:512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p:.4' \
                       '->convsh|f: 512,r:3,l2_val:5e-4,bias:1->relu->bn->dropout|p: .4' \
                       '->convsh|f: 512,r: 3,l2_val:5e-4,bias:1->relu->bn' \
                       '->maxpool|r:2,s:2->dropout|p:.5' \
                       '->convsh|f: 512,r: 1,l2_val:5e-4,bias:1->relu->bn->dropout|p:.5' \
                       '->convshfixedfilter|f:{},r:1,bias:0' \
                       '->flattensh->softmax->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
################## Christmass 18 Experiments

def nin_baseline2_ifc_bn_snsrecv2_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrec|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrec|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrec|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrec|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

def nin_baseline2_ifc_bn_snsrecv2shratio(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratio|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrecshratio|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratio|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrecshratio|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_snsrecv2shratiorev(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratiorev|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrecshratiorev|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratiorev|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrecshratiorev|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline2_ifc_bn_snsrecv2shratio1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratio|f:160,r:3,l2_val:5e-4,bias:0,shratio:.8->bn->ber' \
                   '->convsnsrecshratio|f:96,r:1,l2_val:5e-4,bias:0,shratio:.8->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratio|f:192,r:1,l2_val:5e-4,bias:0,shratio:.8->bn->ber' \
                   '->convsnsrecshratio|f:192,r:1,l2_val:5e-4,bias:0,shratio:.8->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_snsrecv2shratiorev1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratiorev|f:160,r:3,l2_val:5e-4,bias:0,shratio:.8->bn->ber' \
                   '->convsnsrecshratiorev|f:96,r:1,l2_val:5e-4,bias:0,shratio:.8->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecshratiorev|f:192,r:1,l2_val:5e-4,bias:0,shratio:.8->bn->ber' \
                   '->convsnsrecshratiorev|f:192,r:1,l2_val:5e-4,bias:0,shratio:.8->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline2_ifc_bn_snsrecv2fixed(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecratiofixed|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrecratiofixed|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsrecratiofixed|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsrecratiofixed|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
## End of Christmas 18 Experiments
## AMir Experiment parallel softmax

def nin_baseline2_ifc_bn_snsdy(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdy|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsdy|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdy|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsdy|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_snsdyrec(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdyrec|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsdyrec|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdyrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsdyrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_bn_snsdyrecv2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdyrecv2|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsdyrecv2|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdyrecv2|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->ber' \
                   '->convsnsdyrecv2|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->ifcv2|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

def nin_baseline2_ifc_bn_quant(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdyrec|f:160,r:3,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->convsnsdyrec|f:96,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsnsdyrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->convsnsdyrec|f:192,r:1,l2_val:5e-4,bias:0,shratio:.5->bn->relu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->ifc|out:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

def nin_baseline2_ifc_bn_sns2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsns|f:160,r:3,l2_val:5e-4,bias:0,shratio:.75->bn->ber' \
                   '->convsns|f:96,r:1,l2_val:5e-4,bias:0,shratio:.75->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsns|f:192,r:1,l2_val:5e-4,bias:0,shratio:.75->bn->ber' \
                   '->convsns|f:192,r:1,l2_val:5e-4,bias:0,shratio:.75->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_dconv_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'dconv|f:192,r:5,l2_val:5e-4,bias:0,iter:1,mult:1->bnsh->relu' \
                   '->convsh|f:160,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->ifconcat|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->dconv|f:192,r:5,l2_val:5e-4,bias:0,iter:4,mult:1->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->ifconcat|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->dconv|f:192,r:3,l2_val:5e-4,bias:0,iter:4,mult:1->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:8,s:1' \
                    '->ifconcat'\
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifc_dconv_concat_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'dconvconcat|f:192,r:5,l2_val:5e-4,bias:0,iter:3,mult:1->bn->relu' \
                   '->convsh|f:160,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->dconvconcat|f:192,r:5,l2_val:5e-4,bias:0,iter:3,mult:1->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->dconvconcat|f:192,r:3,l2_val:5e-4,bias:0,iter:3,mult:1->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->ifconcat' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

def nin_baseline2_ifconcat_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:24,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:24,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifconcat2_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:12,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:24,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifconcat2_bec_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:48,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnr|f:48,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:48,r:1,l2_val:5e-4,bias:0' \
                   '->ifconcat' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:48,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnr|f:48,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:48,r:1,l2_val:5e-4,bias:0' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifconcat2_bec_bn2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnr|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->ifconcat' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:96,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnr|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:96,r:1,l2_val:5e-4,bias:0' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifconcat2_bec_bn3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->becbnr|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:8,s:1' \
                   '->ifconcat' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifconcat2_bec_bn4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnr|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->becbnr|f:192,r:5,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->becbnr|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->becbnr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline2_ifconcat2_bec_bnsh4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->relu' \
                   '->becbnshr|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->becbnshr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->becbnshr|f:192,r:5,l2_val:5e-4,bias:0' \
                   '->becbnshr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->becbnshr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->becbnshr|f:192,r:3,l2_val:5e-4,bias:0' \
                   '->becbnshr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->becbnshr|f:192,r:1,l2_val:5e-4,bias:0' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
def nin_baseline2_ifconcat_resnet(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:48,r:1,l2_val:5e-4,bias:0->bn->push|name:a' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:80,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:48,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->mask|name:a' \
                   '->ifconcat|out:1' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bn' \
                   '->maxpool|r:3,s:2,pad:same->push|name:b' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->mask|name:b' \
                   '->ifconcat|out:1' \
                   '->convsh|f:96,r:5,l2_val:5e-4,bias:0->bn' \
                   '->averagepool|r:3,s:2,pad:same->push|name:c' \
                   '->convsh|f:96,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->mask|name:c' \
                   '->ifconcat|out:1' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_bnsh(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:7,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_bnsh_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:1->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:1->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->relu' \
                   '->averagepool|r:7,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_bnsh_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'bnsh->convsh|f:192,r:5,l2_val:5e-4,bias:1->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:1->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:1->ber' \
                   '->ifc|out:1->bnsh' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:1->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->ber' \
                   '->ifc|out:1->bnsh' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:1->relu' \
                   '->averagepool|r:7,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_bnsh2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4,bias:0->bnsh->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4,bias:0->bnsh->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4,bias:0->bnsh->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4,bias:0->bnsh->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bnsh->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bnsh->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:7,s:1' \
                   '->convsh|f:{},r: 1,l2_val:1e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:5e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:5e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4->relu' \
                   '->convsh|f:{},r:1,l2_val:5e-4->relu' \
                   '->averagepool|r:7,s:1' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_bn(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:7,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_ifc_bn2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4,bias:0->bn->ber' \
                   '->convsh|f:160,r:3,l2_val:1e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:1e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bn->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bn->ber' \
                   '->ifc|out:1' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4,bias:0->bn->relu' \
                   '->averagepool|r:7,s:1' \
                   '->convsh|f:{},r: 1,l2_val:1e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)



def nin_branch1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:160,r:3,l2_val:1e-4->relu' \
                   '->convsh|f:96,r:1,l2_val:1e-4->relu' \
                   '->maxpool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:5,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->convsh|f:192,r:1,l2_val:1e-4->relu' \
                   '->averagepool|r:3,s:2->dropout|p:.5' \
                   '->convsh|f:192,r:3,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->convsh|f:192,r:1,l2_val:1e-4->ber' \
                   '->averagepool|r:7,s:1->dropout|p:.5' \
                   '->conv|f:4,r:1,l2_val:1e-4->sigmoid' \
                   '->concat' \
                   '->flattensh->densesh|n:{}' \
                   '->softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)

# DECONV
def nin_baseline2_ifc_dconv_concat_bn_1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'dconvberconcat|f:192,r:3,l2_val:5e-4,bias:0,iter:3,mult:1->bn->relu' \
                   '->convsh|f:160,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->dconvberconcat|f:192,r:3,l2_val:5e-4,bias:0,iter:3,mult:1->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->ifconcat|out:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->dconvberconcat|f:192,r:3,l2_val:5e-4,bias:0,iter:3,mult:1->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn->relu' \
                   '->averagepool|r:8,s:1' \
                   '->ifconcat' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bn_conv_ber(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->convsh|f:80,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:48,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat|out:1->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat|out:1->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->ifconcat|out:1->convsh|f:192,r:1,l2_val:5e-4,bias:0->bn' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)
# AMP
def nin_baseline_bn_adarelu(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ber' \
                   '->conv|f:80,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->adarelu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:192,r:1,l2_val:5e-4,bias:0->bn->adarelu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->ber' \
                   '->conv|f:192,r:1,l2_val:5e-4,bias:0->bn->adarelu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bn_ampadarelu(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ampber' \
                   '->conv|f:80,r:3,l2_val:5e-4,bias:0->bn->ampber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->adarelu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bn->ampber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->conv|f:192,r:1,l2_val:5e-4,bias:0->bn->adarelu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bn->ampber' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->conv|f:192,r:1,l2_val:5e-4,bias:0->bn->adarelu' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bn_ampconv_ampber(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ampber' \
                   '->ampconv|f:80,r:3,l2_val:5e-4,bias:0->bn->ampber' \
                   '->ampconv|f:96,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->imean'\
                   '->maxpool|r:3,s:2,pad:same' \
                   '->ampconv|f:192,r:5,l2_val:5e-4,bias:0->bn->ampber' \
                   '->ampconv|f:96,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->imean' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->ampconv|f:192,r:3,l2_val:5e-4,bias:0->bn->ampber' \
                   '->ampconv|f:96,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->ampber' \
                   '->imean' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bn_amp_l1(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->amprelu|norm:$norm' \
                   '->ampconv|f:160,r:3,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:96,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->ampconv|f:192,r:5,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->ampconv|f:192,r:3,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes).replace('$norm','1')

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bn_amp_l2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->amprelu|norm:$norm' \
                   '->ampconv|f:160,r:3,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:96,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->ampconv|f:192,r:5,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->ampconv|f:192,r:3,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->ampconv|f:192,r:1,l2_val:5e-4,bias:0->bn->amprelu|norm:$norm' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes).replace('$norm', '2')

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bn_ampbimean_l2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->ampbmeanrelu|norm:$norm' \
                   '->$conv|f:160,r:3,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->$conv|f:96,r:1,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->$conv|f:192,r:5,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->$conv|f:192,r:1,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->$conv|f:192,r:1,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->$conv|f:192,r:3,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->$conv|f:192,r:1,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->$conv|f:192,r:1,l2_val:5e-4,bias:0,norm:$norm->bn->ampbmeanrelu|norm:$norm' \
                   '->averagepool|r:8,s:1' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes).replace('$norm', '2').replace('$conv','convsh')

    return get_model_out_dict(opts, model_string=model_string)

# split
def nin_split(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:80,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->ifconcat' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->ifconcat' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                    '->ifconcat'\
                   '->averagepool|r:8,s:1' \
                   '->conv|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_split2(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:80,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:96,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->conv|f:48,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->conv|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_split3(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:80,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                    '->ifc|out:2'\
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:96,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:4' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->conv|f:48,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->conv|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_split4(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:160,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:96,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:2' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->conv|f:96,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:2' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:48,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->conv|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_split5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:8' \
                   '->conv|f:20,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:12,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:24,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:8' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:2' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->conv|f:96,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:4' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->conv|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_split5(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'conv|f:192,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:8' \
                   '->conv|f:20,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:12,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:1' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->conv|f:24,r:5,l2_val:5e-4,bias:0->bn->relusplit|child:8' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifc|out:2' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->conv|f:96,r:3,l2_val:5e-4,bias:0->bn->relusplit|child:4' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->conv|f:24,r:1,l2_val:5e-4,bias:0->bn->relusplit|child:1' \
                   '->ifconcat' \
                   '->averagepool|r:8,s:1' \
                   '->conv|f:{},r: 1,l2_val:5e-4' \
                   '->flattensh->' \
                   'softmax' \
                   '->fin'.format(nb_classes)

    return get_model_out_dict(opts, model_string=model_string)


def nin_baseline_bnsh_psoft(opts, input_shape, nb_classes, getstring_flag=False):
    # softmax then average
    model_string = 'convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:160,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:96,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->maxpool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:5,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->averagepool|r:3,s:2,pad:same' \
                   '->convsh|f:192,r:3,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:192,r:1,l2_val:5e-4,bias:0->bnsh->relu' \
                   '->convsh|f:{},r: 1,l2_val:5e-4' \
                   '->psoftmax' \
                   '->averagepool|r:8,s:1' \
                   '->flattensh' \
                   '->fin'.format(nb_classes)
    return get_model_out_dict(opts, model_string=model_string)
def get(identifier):
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret model identifier:', identifier)


def serialize(initializer):
    return serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config, module_objects=globals(), custom_objects=custom_objects, printable_module_name='model')


def get_model_from_db(identifier, opts):
    model_fun = get(globals()[identifier])
    dict = model_fun(opts, opt_utils.get_input_shape(opts), opt_utils.get_nb_classes(opts))

    return dict


def get_model_dict_from_db(identifier, opts):
    model_fun = get(globals()[identifier])
    dict = model_fun(opts, opt_utils.get_input_shape(opts), opt_utils.get_nb_classes(opts))

    return dict


if __name__ == '__main__':
    opts = default_opt_creator()
    functions = globals().copy()
    func_to_test = ['max_entropy_perm_nin_model1', 'baseline2_besh12', 'besh_crelu_12']
    for function in functions:
        if function not in func_to_test:
            continue
        model = functions.get(function)
        print(function)
        opts = opt_utils.set_model_string(opts, function)
        opts = opt_utils.set_dataset(opts, 'cifar100')
        opt_utils.set_default_opts_based_on_model_dataset(opts)
        model = model(opts, (3, 32, 32), 10)
        model.summary()
    for function in functions:
        if not function in func_to_test:
            continue
        model = functions.get(function)
        print(function)
        opts = opt_utils.set_model_string(opts, function)
        opts = opt_utils.set_dataset(opts, 'cifar100')
        opt_utils.set_default_opts_based_on_model_dataset(opts)
        model = model(opts, (3, 32, 32), 10)
        print((model.count_params()))
