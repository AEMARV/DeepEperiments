from modeldatabase.Binary_models.model_db import get_model_out_dict
from utils.modelutils.layers.kldivg.initializers import *
from utils.modelutils.layers.kldivg.regularizers import *
from utils.modelutils.layers.kldivg.distances import *
from utils.modelutils.layers.kldivg.optimizers import *
def MLP_v0(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:3000,r:32,l2_val:5e-4,bias:0,padding:valid,isrelu:0' \
                   '->mixer|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    coef = 1
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func, linker=linker, coef = coef)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker, coef =coef)
    opts['model_opts']['kl_opts']['mixer_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker,coef = coef)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.0
    opts['optimizer_opts']['lr'] = 1 # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def MLP_v0(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:2000,r:32,l2_val:5e-4,bias:0,padding:valid,isrelu:1' \
                   '->mixer|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    coef = 1
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func, linker=linker, coef = coef)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker, coef =coef)
    opts['model_opts']['kl_opts']['mixer_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker,coef = coef)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.0
    opts['optimizer_opts']['lr'] = 1 # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)