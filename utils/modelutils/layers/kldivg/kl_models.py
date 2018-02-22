from modeldatabase.Binary_models.model_db import get_model_out_dict
from utils.modelutils.layers.kldivg.initializers import *
from utils.modelutils.layers.kldivg.regularizers import *
from utils.modelutils.layers.kldivg.distances import *
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
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_model_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Sigmoid_Init()
    opts['model_opts']['kl_opts']['kl_initial'] = Softmax_Init()
    opts['model_opts']['kl_opts']['dist_measure'] = kl_v1
    opts['model_opts']['kl_opts']['use_link_func'] = True
    return get_model_out_dict(opts, model_string=model_string)
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
