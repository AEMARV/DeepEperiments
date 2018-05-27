from modeldatabase.Binary_models.model_db import get_model_out_dict
from utils.modelutils.layers.kldivg.initializers import *
from utils.modelutils.layers.kldivg.regularizers import *
from utils.modelutils.layers.kldivg.distances import *
from utils.modelutils.layers.kldivg.optimizers import *
def kl_nin_baseline_sqnat_Final(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'klconvb|f:192,r:5,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:160,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:96,r:1,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:3,s:2,pad:same' \
                   '->klconv|f:192,r:5,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:3,s:2,pad:same' \
                   '->klconv|f:192,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:8,s:1' \
                   '->klconv|f:{},r:1,bias:0,isrelu:1->lsoft' \
                   '->flattensh->fin'.format(nb_classes)
    use_link_func = True
    linker = linker_square
    initCoef = 4

    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def kl_nin_baseline_logsimp_Final(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'klconvb|f:192,r:5,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:160,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:96,r:1,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:3,s:2,pad:same' \
                   '->klconv|f:192,r:5,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:3,s:2,pad:same' \
                   '->klconv|f:192,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:192,r:1,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:8,s:1' \
                   '->klconv|f:{},r:1,bias:0,isrelu:1->lsoft' \
                   '->flattensh->fin'.format(nb_classes)

    use_link_func = False
    linker = linker_square
    initCoef = 4
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['bias_initial'] = Dirichlet_Init_Bias(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def kl_vgg_baseline_sqnat_Final(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'klconvb|f:64,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:64,r:3,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:2,s:2' \
                   '->klconv|f:128,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:128,r:3,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:2,s:2' \
                   '->klconv|f:256,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:256,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:256,r:3,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:2,s:2' \
                   '->klconv|f:512,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:512,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f:512,r:3,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:2,s:2' \
                   '->klconv|f:512,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f: 512,r:3,bias:0,isrelu:1->lsoft' \
                   '->klconv|f: 512,r:3,bias:0,isrelu:1->lsoft' \
                   '->klavgpool|r:2,s:2' \
                   '->klconv|f: 512,r: 1,bias:1,isrelu:1->lsoft' \
                   '->klconv|f:{},r:1,bias:0,isrelu:1->lsoft' \
                   '->flattensh->fin'.format(nb_classes)
    use_link_func = True
    linker = linker_square
    initCoef = 8

    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
'''NIPS FINAL HELLO KL'''
def hello_kl_final_logsimp(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = False
    linker = linker_square
    initCoef = 2
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['bias_initial'] = Dirichlet_Init_Bias(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = False
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
'''Log-Simplex Parametrization'''
def helloKl_loginit(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = LogSimplexParSphericalInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = LogSimplexParSphericalInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = LogSimplexParSphericalInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum= opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_loginit_dirichlet(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = Dirichlet_Init(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = Dirichlet_Init_Bias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum= opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
''' L2-Simplex Parametrization Experiments'''
def helloKl_nat_sq_v0(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1,isrelu:0->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.0
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v1(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1,isrelu:0->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)


    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v2(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.0
    opts['optimizer_opts']['lr'] = 100  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v3(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.0
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.0,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v4(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v5(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v6(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v7(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v8(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None # Low bound Entropy
    opts['model_opts']['kl_opts']['convreg'] = None# Low bound Entropy
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_v9(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None # Low bound Entropy
    opts['model_opts']['kl_opts']['convreg'] = None# Low bound Entropy
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
'''Full sigmoid Experiments'''
def helloKl_Sigmoidal_v0(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1,isrelu:0->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1,isrelu:0->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1,isrelu:0->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1,isrelu:0->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:0->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
'''Mixture Experiments'''
def helloKl_nat_sq_mixerv0(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->mixer|f:32->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->mixer|f:64->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->mixer|f:128->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:1->mixer|f:192->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_nat_sq_mixerv1(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    use_link_func = True
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:1' \
                   '->mixer|f:' + str(nb_classes) + '->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)

def helloKl_nat_sq_mixerv2(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:0' \
                   '->mixer|f:' + str(nb_classes) + '->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_mixerv3(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    use_link_func = True
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->mixer|f:64->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:1' \
                   '->mixer|f:' + str(nb_classes) + '->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_mixerv4(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:1->mixer|f:16->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:1->mixer|f:32->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:1->mixer|f:32->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:1->mixer|f:64->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)
def helloKl_nat_sq_mixerv5(opts, input_shape, nb_classes, getstring_flag=False):
    # Same Structure as nin besh 1 2 3
    regklb = None
    distvec=[]
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:0,isrelu:0->mixer|f:16->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:0,isrelu:0->mixer|f:32->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:0,isrelu:0->mixer|f:32->lsoft|reg:0' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:0,isrelu:0->mixer|f:64->lsoft|reg:0' \
                   '->klconv|f:' + str(nb_classes) + ',r:1,isrelu:1->lsoft|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = True
    linker = linker_square
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = UnitSphereInitBin(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['kl_initial'] = UnitSphereInit(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['bias_initial'] = UnitSphereInitBias(use_link_func=use_link_func, linker=linker)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.9,
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)

'''Stochastic Param'''
'''NIPS FINAL HELLO KL'''
def hello_kl_stoch(opts, input_shape, nb_classes, getstring_flag=False):
    model_string = 'klconvb|f:32,r:5,l2_val:5e-4,bias:1->lsofts' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:64,r:5,l2_val:1e-4,bias:1->lsofts' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:128,r:3,l2_val:1e-4,bias:1->lsofts' \
                   '->klavgpool|r:3,s:2' \
                   '->klconv|f:192,r:1,l2_val:1e-4,bias:1->lsofts' \
                   '->klconv|f:' + str(nb_classes) + ',r:1->lsofts|reg:0' \
                                                     '->klavgpool|r:3,s:1' \
                                                     '->flattensh' \
                                                     '->fin'
    use_link_func = False
    linker = linker_square
    initCoef = 2
    opts['model_opts']['kl_opts'] = {}
    opts['optimizer_opts']['loss']['method'] = kl_loss_data_centric
    opts['model_opts']['kl_opts']['convbreg'] = None
    opts['model_opts']['kl_opts']['convreg'] = None
    opts['model_opts']['kl_opts']['klb_initial'] = Dirichlet_Init_Bin(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['kl_initial'] = Stoch_Param(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['bias_initial'] = Dirichlet_Init_Bias(use_link_func=use_link_func, linker=linker, coef=initCoef)
    opts['model_opts']['kl_opts']['dist_measure'] = kl_both_centric
    opts['model_opts']['kl_opts']['use_link_func'] = use_link_func
    opts['model_opts']['kl_opts']['biasreg'] = None
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lrchange'] = False
    opts['optimizer_opts']['lr'] = 1  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=opts['optimizer_opts']['momentum'],
                                                   decay=0.0,
                                                   nesterov=False)
    return get_model_out_dict(opts, model_string=model_string)