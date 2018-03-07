from modeldatabase.Binary_models.model_db import get_model_out_dict
from utils.modelutils.layers.kldivg.initializers import *
from utils.modelutils.layers.kldivg.regularizers import *
from utils.modelutils.layers.kldivg.distances import *
'''Hello KL variants'''
# Base Model
def simplenn_BE_lsoft(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'convsh|f:32,r:5,l2_val:0->lsoft' \
                   '->maxpool|r:3,s:2' \
                   '->convsh|f:64,r:5,l2_val:0->lsoft' \
                   '->averagepool|r:3,s:2' \
                   '->convsh|f:128,r:3,l2_val:0->lsoft' \
                   '->averagepool|r:3,s:2' \
                   '->convsh|f:192,r:1,l2_val:0->lsoft' \
                   '->convsh|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->averagepool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->softmax->fin'
    return get_model_out_dict(opts, model_string=model_string)

def helloKl(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_LinkFunc_Spherical(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = True
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Unit_Sphere_Init_Bin(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Unit_Sphere_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

# Loss and KL assymetry experiments
def helloKl_datacentric(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_data_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_modelcentric(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_model_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_both_centric_both_layersandloss(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_both_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_data_loss_both_centric(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_both_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_data_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_model_loss_both_centric(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_both_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_model_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_data_loss_model(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_data_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_model_loss_data(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_data_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

#Cross Entropy Models
def helloKl_layers_cross_model_centric_loss_data(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_cross_ent_model_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_cross_data_centric_loss_data(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_cross_ent_data_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_cross_model_centric_loss_model(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_cross_ent_model_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_layers_cross_data_centric_loss_model(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_cross_ent_data_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)
# Unit Sphere Models
def helloK_Jeffreys(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvbSP|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = True
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Unit_Sphere_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Unit_Sphere_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloK_Jeffreys_data_centric_dist(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvbSP|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['model_opts']['kl_opts'] = {}
    use_link_func = True
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Unit_Sphere_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Unit_Sphere_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_model_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

# Unnormalized
def helloKl_UnNorm_Spherical(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvbu|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvu|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvu|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvu|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconvu|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = True
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Unit_Sphere_Init_Bin(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Unit_Sphere_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)
# Others
def helloKl_super_small(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:10,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:10,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:10,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:10,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init()
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init()
    opts['model_opts']['kl_opts']['dist_measure'] = kl_v3
    opts['model_opts']['kl_opts']['use_link_func'] = True
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_reg(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,coef:1e-5->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,coef:1e-5->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,coef:1e-5->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,coef:1e-5->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,coef:1e-5->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = Ent_Reg_Sigmoid
    opts['model_opts']['kl_opts']['convreg'] = Ent_Reg_Softmax
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init()
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init()
    opts['model_opts']['kl_opts']['dist_measure'] = kl_v3
    opts['model_opts']['kl_opts']['use_link_func'] = True
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_inc_firstlay(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:128,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init()
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init()
    opts['model_opts']['kl_opts']['dist_measure'] = kl_v1
    opts['model_opts']['kl_opts']['use_link_func'] = True
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_regv1(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:128,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init()
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init()
    opts['model_opts']['kl_opts']['dist_measure'] = kl_v1
    opts['model_opts']['kl_opts']['use_link_func'] = True
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_DistKerfromX_NoReg(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:42,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:57,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:114,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:116,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_both_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init()
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init()
    opts['model_opts']['kl_opts']['dist_measure'] = kl_v1
    opts['model_opts']['kl_opts']['use_link_func'] = True
    return get_model_out_dict(opts, model_string=model_string)

# Bregman KL divg
def helloKlBreg(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvbreg|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvbreg|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvbreg|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconvbreg|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_bregman
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    return get_model_out_dict(opts, model_string=model_string)

def helloKlBreg_UnNorm(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvbSPbreg|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvbregunnorm|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvbregunnorm|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvbregunnorm|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconvbregunnorm|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_bregman
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Exp_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Exp_Init_Norm(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    ## Optimizer Opts
    opts['optimizer_opts']['momentum'] = 0.9
    return get_model_out_dict(opts, model_string=model_string)

# Concentrated KL divg
def helloKl_Concentrated(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvbconc|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvconc|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvconc|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconvconc|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconvconc|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    use_link_func = False
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] =Sigmoid_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    ## Optimizer Opts
    opts['optimizer_opts']['momentum'] = 0.9
    return get_model_out_dict(opts, model_string=model_string)

# NIN Variants
def nin_KL(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'klconvb|f:192,r:5->lsoft' \
                   '->klconv|f:160,r:1->lsoft' \
                   '->klconv|f:96,r:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:5->lsoft' \
                   '->klconv|f:192,r:1->lsoft' \
                   '->klconv|f:192,r:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:3->lsoft' \
                   '->klconv|f:192,r:1->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:7,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'

    return get_model_out_dict(opts, model_string=model_string)

# Deep Models
def kldeeperv1(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4->lsoft' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klconv|f:64,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4->lsoft' \
                   '->klconv|f:128,r:5,l2_val:1e-4->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:192,r:1,l2_val:1e-4->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->lsoft->fin'
    opts['optimizer_opts']['loss']['method'] = kl_loss
    return get_model_out_dict(opts, model_string=model_string)

