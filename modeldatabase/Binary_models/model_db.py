import six
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.utils import serialize_keras_object, deserialize_keras_object

from modeldatabase.Binary_models.model_constructor_utils import get_model, node_list_to_list
from utils import opt_utils
from utils.opt_utils import default_opt_creator


def Layer_on_list(layer, tensor_list):
	res = []
	tensor_list = node_list_to_list(tensor_list)
	for x in tensor_list:
		res += [layer(x)]
	return res


def simplenn_BE(opts, input_shape, nb_classes, getstring_flag=False):
	# Same Structure as nin besh 1 2 3
	model_string = 'convsh|f:32,r:5,l2_val:1e-4->ber' \
	               '->maxpool|r:3,s:2->dropout|p:.5' \
	               '->convsh|f:64,r:5,l2_val:1e-4->relu' \
	               '->averagepool|r:3,s:2->dropout|p:.5' \
	               '->convsh|f:128,r:3,l2_val:1e-4->relu' \
	               '->averagepool|r:3,s:2->dropout|p:.5' \
	               '->convsh|f:192,r:1,l2_val:1e-4->relu' \
	               '->convsh|f:' + str(nb_classes) + ',r:1->relu' \
	                                                 '->averagepool|r:3,s:1' \
	                                                 '->flattensh->merge_branch_average' \
	                                                 '->softmax->fin'
	return get_model(opts, model_string=model_string)


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

	return get_model(opts, model_string=model_string)


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

	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)

""""Structure Test"""
def nin_tree_structure_1(opts, input_shape, nb_classes, getstring_flag=False):
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
	               'softmax->weighted_average_pred_1' \
	               '->fin'.format(nb_classes)
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)

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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)

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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)

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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)
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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	return get_model(opts, model_string=model_string)


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
	model = get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list)
	return model


def besh_vgg_baseline(opts, input_shape, nb_classes, getstring_flag=False):
	model = VGG16(False, input_shape=input_shape)
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(4096, activation='relu'))
	model.add(Dense(nb_classes, activation='softmax'))
	return model


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
	return model_fun(opts, opt_utils.get_input_shape(opts), opt_utils.get_nb_classes(opts))




if __name__ == '__main__':
	opts = default_opt_creator()
	functions = globals().copy()
	func_to_test = ['max_entropy_perm_nin_model1', 'baseline2_besh12', 'besh_crelu_12']
	for function in functions:
		if function not in func_to_test:
			continue
		model = functions.get(function)
		print function
		opts = opt_utils.set_model_string(opts, function)
		opts = opt_utils.set_dataset(opts, 'cifar100')
		opt_utils.set_default_opts_based_on_model_dataset(opts)
		model = model(opts, (3, 32, 32), 10)
		model.summary()
	for function in functions:
		if not function in func_to_test:
			continue
		model = functions.get(function)
		print function
		opts = opt_utils.set_model_string(opts, function)
		opts = opt_utils.set_dataset(opts, 'cifar100')
		opt_utils.set_default_opts_based_on_model_dataset(opts)
		model = model(opts, (3, 32, 32), 10)
		print model.count_params()
