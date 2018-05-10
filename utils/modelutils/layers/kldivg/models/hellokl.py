from modeldatabase.Binary_models.model_db import get_model_out_dict
from utils.modelutils.layers.kldivg.initializers import *
from utils.modelutils.layers.kldivg.regularizers import *
from utils.modelutils.layers.kldivg.distances import *
from utils.modelutils.layers.kldivg.optimizers import *
'''Log-Simplex Parametrization'''
def helloKl(opts, input_shape, nb_classes, getstring_flag=False):
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
''' L2-Simplex Parametrization Experiments'''
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
    opts['optimizer_opts']['lr'] = 100  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.0,
                                                   decay=0.0,
                                                   nesterov=False)
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
    opts['optimizer_opts']['momentum'] = 0.9
    opts['optimizer_opts']['lr'] = 100  # .1
    opts['optimizer_opts']['optimizer'] = PolarSGD(polar_decay=0,
                                                   lr=opts['optimizer_opts']['lr'],
                                                   momentum=0.0,
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
    opts['optimizer_opts']['momentum'] = 0.9
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
                                                   momentum=0.9,
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