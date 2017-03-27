from Models.Binary_models.model_utils import get_model
from keras.utils.generic_utils import get_from_module
from utils import opt_utils
from utils.opt_utils import default_opt_creator

def be0(opts, input_shape, nb_classes,getstring_flag=False):
	'eeee'
	model_string = 'e|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->e|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->e|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def bes0(opts, input_shape, nb_classes,getstring_flag=False):
	#53.01%
	'eess'
	model_string = 'e|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->s|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]

	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def besm0(opts, input_shape, nb_classes,getstring_flag=False):
	'eesrm'
	#53.82%
	model_string = 'e|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def besm1(opts, input_shape, nb_classes,getstring_flag=False):
	'eemm'
	# 52.6
	model_string = 'e->mp->e->ap->rm->ap->rm'
	model_string = 'e|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->bm|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,2]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def besm2(opts, input_shape, nb_classes,getstring_flag=False):
	'essm'
	#52.33
	model_string = 'e->mp->s->ap->s->ap->rm->ap'
	model_string = 'e|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->bm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0(opts, input_shape, nb_classes,getstring_flag=False):
	#56%
	model_string = 'e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_0(opts, input_shape, nb_classes,getstring_flag=False):
	#53.5
	model_string = 'e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->am|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_1(opts, input_shape, nb_classes,getstring_flag=False):
	#49.72
	model_string = 'e|f:32,r:5' \
	               '->s|f:64,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:256,r:5' \
	               '->ap|s:2,r:3' \
	               '->bm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil1_0(opts, input_shape, nb_classes,getstring_flag=False):
	#49.88
	model_string = 'e|f:32,r:5' \
	               '->e|f:64,r:5' \
	               '->e|f:128,r:5' \
	               '->bm|f:256,r:5' \
	               '->bm|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->bm|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->rm|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->r|f:256,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_2(opts, input_shape, nb_classes,getstring_flag=False):
	#51.82
	model_string = 'e|f:32,r:5' \
	               '->e|f:64,r:5' \
					'->e|f:128,r:5' \
	               '->s|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->bm|f:128,r:4' \
					'->bm|f:128,r:4' \
					'->rm|f:128,r:4'\
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3(opts, input_shape, nb_classes,getstring_flag=False):
	# not learning
	model_string = 'e|f:32,r:5' \
	               '->e|f:64,r:5' \
	               '->e|f:128,r:5' \
	               '->s|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->s|f:256,r:4' \
	               '->ap|s:2,r:3'\
	               '->bm|f:256,r:4' \
	               '->bm|f:128,r:4' \
	               '->bm|f:64,r:4'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_0(opts, input_shape, nb_classes, getstring_flag=False):
	# val_err:54.70%
	model_string = 'e|f:32,r:5' \
	               '->e|f:64,r:5' \
	               '->e|f:128,r:5' \
	               '->s|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->s|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_1(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->ap|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->am|f:128,r:5' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->am|f:512,r:4' \
	               '->ap|s:2,r:3' \


	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_4(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->ap|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->e|f:256,r:5' \
	               '->s|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->s|f:256,r:4' \
	               '->ap|s:2,r:3' \


	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_5(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->ap|s:2,r:3' \
	               '->e|f:64,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->e|f:256,r:5' \
	               '->s|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->am|f:512,r:5' \
	               '->s|f:256,r:4' \
	               '->ap|s:2,r:3' \


	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_6(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->e|f:64,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->am|f:256,r:5' \
	               '->s|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->e|f:256,r:5' \
	               '->s|f:128,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_0_1(opts, input_shape, nb_classes, getstring_flag=False):
	# val_Err : 54% param :5e5
	model_string = 'e|f:32,r:5' \
	               '->e|f:64,r:5' \
	               '->e|f:128,r:5' \
	               '->s|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->a|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e(opts, input_shape, nb_classes,getstring_flag=False):
	# val_err : 45% param : 919428
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'e|f:32,r:5' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->am|f:64,r:4' \
	               '->am|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0(opts, input_shape, nb_classes,getstring_flag=False):
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'e|f:32,r:5' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0(opts, input_shape, nb_classes,getstring_flag=False):
	# val: 45% epoch 48
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'e|f:32,r:5' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0_rb(opts, input_shape, nb_classes,getstring_flag=False):
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.5' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_rb0(opts, input_shape, nb_classes,getstring_flag=False):
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.1' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_rb1(opts, input_shape, nb_classes,getstring_flag=False):
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.25' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_rb2(opts, input_shape, nb_classes,getstring_flag=False):
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0_rb0(opts, input_shape, nb_classes,getstring_flag=False):
	# 48.21
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:0' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0_rb1(opts, input_shape, nb_classes,getstring_flag=False):
	# 50.18 %
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.25' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0_rb2(opts, input_shape, nb_classes,getstring_flag=False):
	#50  %
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.5' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0_rb3(opts, input_shape, nb_classes,getstring_flag=False):
	# 51.2%
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_e_0_rb4(opts, input_shape, nb_classes,getstring_flag=False):
	# 48.87
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:1' \
	               '->e|f:32,r:5' \
	               '->s|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_0_rb0(opts, input_shape, nb_classes, getstring_flag=False):
	# val : 56.6
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:64,r:5' \
	               '->rbe|f:128,r:5' \
	               '->s|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->s|f:256,r:4' \
	               '->ap|s:2,r:3'

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def be0_rb0(opts, input_shape, nb_classes,getstring_flag=False):
	'eeee'
	# 52.5 could be better if more epochs. Hypothesis: if we increase learning rate not gonna affect our
	# 53.52 param:1.7e6
	# 54.39 param;2.5e6
	# 55.30 param:5.6e6


	# learning
	# this was based on observation
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def be1_rb0(opts, input_shape, nb_classes,getstring_flag=False):
	'eeee'
	#52.49 param 9.8e5
	#54.38 param 1.7 e6
	# 55.83 param:3.8 e6
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:64,r:3' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def be2_rb0(opts, input_shape, nb_classes,getstring_flag=False):
	# probably is going to beat the baseline. 31.73% till epoch 26
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:64,r:5' \
	               '->rbe|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_rb4(opts, input_shape, nb_classes,getstring_flag=False):
	# lost hope 52.14
	model_string  = 'e->s->mp->s->ap->s->ap->rm->ap'
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbs|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbs|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_0_rb1(opts, input_shape, nb_classes, getstring_flag=False):
	# havent finished testing it. lots of parameters at epoch 20 we got 30% which sounds promising but with 8e6
	# parameters
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:64,r:5' \
	               '->rbe|f:128,r:5' \
	               '->rbs|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbs|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_3_l5_rb1(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbs|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_l4_rb0(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbs|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def be0_rb0_ad(opts, input_shape, nb_classes,getstring_flag=False): # ad means adaptive dropout
	# tested the ad it didnt work with cosine and also sigmoid function almost identical to p .75
	model_string = 'rbe|f:32,r:5,p:-1' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
### Experiments 2: these neteworks have same total filters at each layer.

def be0_rb0_fixedfilter_t(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'e|f:4,r:5,p:.75' \
	               '->e|f:4,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->ap|s:2,r:3' \
	               '->ap|s:2,r:3' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_expanded_len(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def lil0_3_l5_rb1(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbs|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_l4_rb0(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->rbe|f:128,r:5' \
	               '->mp|s:2,r:3' \
	               '->rbs|f:256,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:512,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbs|f:256,r:4' \
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def lil0_dropout1(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:32,r:5'\
	               '->mp|s:2,r:3' \
	               '->s|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rm|f:64,r:4' \
	               '->d|p:.2' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
	          conv_filter_size_list=filter_size_list)
def be0_rb0_fixedfilter_dropout2(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:5' \
	               '->d|p:.2' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->d|p:0.25' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_fixedfilter_dropout1(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->d|p:0.25' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_fixedfilter_av(opts, input_shape, nb_classes,getstring_flag=False):
	nb_filter_list = [32,64,128,64]
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->rbe|f:64,r:3' \
	               '->rbe|f:64,r:5' \
	               '->rbe|f:64,r:4' \
	               '->mp|s:2,r:3' \
	               '->ap|s:2,r:3' \
	               '->ap|s:2,r:3' \
	               '->ap|s:2,r:3'
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_fixedfilter_c0(opts, input_shape, nb_classes,getstring_flag=False):
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:3,p:.75' \
	               '->ap|s:2,r:3' \
	               '->c|n:-1' \
	               '->rbe|f:64,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4,p:.75' \
	               '->ap|s:2,r:3'\

	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_fixedfilter_c1(opts, input_shape, nb_classes,getstring_flag=False):
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:3,p:.75' \
	               '->ap|s:2,r:3' \
	               '->c|n:1' \
	               '->rbe|f:64,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4,p:.75' \
	               '->ap|s:2,r:3'\

	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_fixedfilter_c2(opts, input_shape, nb_classes,getstring_flag=False):
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:3,p:.75' \
	               '->ap|s:2,r:3' \
	               '->c|n:1' \
	               '->rbe|f:64,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->c|n:1' \
	               '->rbe|f:64,r:4,p:.75' \
	               '->ap|s:2,r:3'\

	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_fixedfilter_n(opts, input_shape, nb_classes,getstring_flag=False):
	# we can change mp to ap to compare results later
	model_string = 'rben|f:3,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rben|f:3,r:3' \
	               '->ap|s:2,r:3' \
	               '->rben|f:3,r:5' \
	               '->ap|s:2,r:3' \
	               '->rben|f:3,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)


def lil0_baseline1(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:64,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:64,r:5' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)


def lil0_baseline2(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:64,r:5' \
	               '->mp|s:2,r:3' \
	               '->r|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:64,r:5' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
######################################## BASELINE Experiment stored in FinalBaselineMar08
def be0_rb0_fixedfilter(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_final(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_final(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_final_d(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def ln(opts, input_shape, nb_classes, getstring_flag=False):

	# we can change mp to ap to compare results later
	model_string = 'r|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->r|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->r|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string)


def lil0_baseline0(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:64,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:64,r:5' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def lil0_baseline0_d(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'e|f:32,r:5' \
	               '->s|f:64,r:5' \
	               '->mp|s:2,r:3' \
	               '->s|f:128,r:3' \
	               '->ap|s:2,r:3' \
	               '->s|f:64,r:5' \
	               '->ap|s:2,r:3' \
	               '->fullydropout|p:.5'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def be0_rb0_fixedfilter_d(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:64,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->fullydropout|p:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def ln_d(opts, input_shape, nb_classes, getstring_flag=False):

	# we can change mp to ap to compare results later
	model_string = 'r|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->r|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->r|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->fullydropout|p:.5'
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string)
# Trying the leaky version and batch norm after this pos_neg permutation should be tested "leak_rate_expMar9"
# experiment

def be0_rb0_leaky_25(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:.25,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_leaky_50(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.90,leak:.5,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rbeg0_leaky_50(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbeg|f:32,r:5,p:.90,leak:.5,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbeg|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbeg|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_leaky_15(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:.15,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_leaky_15_bn(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:.15,bn:1' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb90_leaky_25(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:90,leak:.25,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb0_leaky_bn_fixed_filter(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:64,r:5,p:.75,leak:.25,bn:1' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
## Leaky Experiment
def be0_rb75_leaky_50(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:.75,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb75_leaky_50_ns(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:.75,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb75_leaky_100(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:1,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb75_leaky_25(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.75,leak:.25,bn:0' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
# Non Symmetric Experiment
def be0_rb80_leaky_15_ns(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.80,leak:.15,bn:0,cp:.75' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def be0_rb60_leaky_0_ns(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'rbe|f:32,r:5,p:.60,leak:0,bn:0,cp:.80' \
	               '->mp|s:2,r:3' \
	               '->rbe|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->rbe|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def prelu_rb75_child80(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'pre|f:32,r:5,p:.75,leak:0,bn:0,cp:.80' \
	               '->mp|s:2,r:3' \
	               '->pre|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->pre|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->pre|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def prelu_rb75_child55(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'pre|f:32,r:5,p:.75,leak:0,bn:0,cp:.60' \
	               '->mp|s:2,r:3' \
	               '->pre|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->pre|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->pre|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def prelu_rb75_child75(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'pre|f:32,r:5,p:.75,leak:0,bn:0,cp:.75' \
	               '->mp|s:2,r:3' \
	               '->pre|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->pre|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->pre|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def prelu_rb75_child50(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'pre|f:32,r:5,p:.75,leak:0,bn:0,cp:.50' \
	               '->mp|s:2,r:3' \
	               '->pre|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->pre|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->pre|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def prelu_rb95_child55(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'pre|f:32,r:5,p:.75,leak:0,bn:0,cp:.55' \
	               '->mp|s:2,r:3' \
	               '->pre|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->pre|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->pre|f:64,r:4' \
	               '->ap|s:2,r:3'\
					'->fullydropout|p:.25'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_0(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->ap|s:2,r:3'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_0_dp75(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:.75' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:4,r:4' \
	               '->ap|s:2,r:3'

	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_0_dp100(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->ap|s:2,r:3'\
				'->fullydropout|p:.50'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_xavr_0_dp100(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'xaesh|f:32,r:5,p:1' \
	               '->ap|s:2,r:3' \
	               '->xaesh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->xaesh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->xaesh|f:64,r:4' \
	               '->ap|s:2,r:3'\
				'->fullydropout|p:.50'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_xavrrelu_0_dp100(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'xaresh|f:32,r:5,p:1' \
	               '->ap|s:2,r:3' \
	               '->xaresh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->xaresh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->xaresh|f:64,r:4' \
	               '->ap|s:2,r:3'\
				'->fullydropout|p:.50'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_xavrrelu_0_dp100_ffully(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->ap|s:2,r:3'\
				'->leaffully|u:1,n:1'\
				'->fullydropout|p:.50'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_0_dp100_ffully(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:2' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4,n:2' \
	               '->ap|s:2,r:3'\
				'->leaffully|u:1,n:1'\
				'->fullydropout|p:.50'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
# Experiments March 13
def besh_crelu_1(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:2' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4,n:2' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_2(opts, input_shape, nb_classes,getstring_flag=False):
	# got semi results from here
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
# Experiment after revelation Mar 13 4:47 AM
# Fully expand Without fully connected
def besh_crelu_3(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
# Deeper model and we want to expand before average in order to sparse the activations
def besh_crelu_4(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:4' \
	               '->mp|s:2,r:3' \
	               '->leaffully|u:1,n:2' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->leaffully|u:1,n:8' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:32' \
	               '->ap|s:2,r:3' \
	               '->leaffully|u:1,n:16' \
	               '->besh|f:64,r:3' \
	               '->shdense|n:-1,do:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
# Adding Maxpool on dense
def besh_crelu_5(opts, input_shape, nb_classes,getstring_flag=False):
	# Added maxpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5,m:1'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_6(opts, input_shape, nb_classes,getstring_flag=False):
	# Added avg of maxpool and avgpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5,m:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_7(opts, input_shape, nb_classes,getstring_flag=False):
	# Added avg of maxpool and avgpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense2|n:-1,do:.5,m:0'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)

def besh_crelu_8(opts, input_shape, nb_classes,getstring_flag=False):
	# Added avg of maxpool and avgpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense2|n:-1,do:.8,m:0'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_10(opts, input_shape, nb_classes,getstring_flag=False):
	# change dropout location of besh_crelu_5
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense3|n:-1,dode:.5,doclas:.5,m:0'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_11(opts, input_shape, nb_classes,getstring_flag=False):
	# c_relu10 with drop out rate .9
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16' \
	               '->ap|s:2,r:3' \
	               '->shdense3|n:-1,dode:.5,doclas:.9,m:0'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_12(opts, input_shape, nb_classes,getstring_flag=False):
	# c_relu10 with drop out rate .9
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2,ido:-1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,doclas:-1,m:0'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
def baseline2_besh12(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->besh|f:64,r:5' \
	               '->leaffully|u:1,n:2,ido:-1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:4,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,doclas:-1,m:0'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def besh_crelu_12(opts, input_shape, nb_classes, getstring_flag=False):
	# c_relu10 with drop out rate .9
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2,ido:-1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16,ido:-1' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,doclas:-1,m:0'
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_13(opts, input_shape, nb_classes, getstring_flag=False):
	# c_relu10 with drop out rate .9
	model_string = 'besh|f:32,r:5,p:1' \
	               '->leaffully|u:1,n:2,chw:1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->leaffully|u:1,n:4,ido:-1,chw:1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->leaffully|u:1,n:8,ido:-1,chw:1' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->leaffully|u:1,n:16,ido:-1,chw:1' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,doclas:-1,m:0'
	nb_filter_list = [32, 64, 128, 64]
	filter_size_list = [5, 3, 3, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_14(opts, input_shape, nb_classes,getstring_flag=False):
	# Added avg of maxpool and avgpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5,m:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_15(opts, input_shape, nb_classes,getstring_flag=False):
	model_string = 'besh|f:32,r:5,p:1' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->shdense|n:-1,do:.5,m:1'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_16(opts, input_shape, nb_classes,getstring_flag=False):
	# Added avg of maxpool and avgpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1,p:.75' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3,p:.75' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4,p:.75' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,m:.5,doclas:.5'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_17(opts, input_shape, nb_classes,getstring_flag=False):
	# Added avg of maxpool and avgpool to besh_crelu_2
	model_string = 'besh|f:32,r:5,p:1,p:.75' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3,p:.75' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4,p:.75' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,m:.5,doclas:.1'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_18(opts, input_shape, nb_classes,getstring_flag=False):
	# crelu_16 with variable droppath
	model_string = 'besh|f:32,r:5,p:1,p:.90' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3,p:.80' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5,p:.75' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4,p:.65' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,m:.5,doclas:.1'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def besh_crelu_19(opts, input_shape, nb_classes,getstring_flag=False):
	# crelu_16 with prob 80%
	model_string = 'besh|f:32,r:5,p:1,p:.80' \
	               '->mp|s:2,r:3' \
	               '->besh|f:64,r:3,p:.80' \
	               '->ap|s:2,r:3' \
	               '->besh|f:128,r:5,p:.80' \
	               '->ap|s:2,r:3' \
	               '->besh|f:64,r:4,p:.80' \
	               '->ap|s:2,r:3' \
	               '->shdensedoi|n:-1,dode:.5,m:.5,doclas:.1'
	nb_filter_list = [32,64,128,64]
	filter_size_list = [5,3,3,3,5,3,4,3]
	if getstring_flag:
		return {'string':model_string,'nb_filter':nb_filter_list,'filter_size':filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string)
def baseline(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'cr|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->cr|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->cr|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->cr|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->d|p:.5'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def baseline_r(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'r|f:32,r:5' \
	               '->mp|s:2,r:3' \
	               '->r|f:64,r:3' \
	               '->ap|s:2,r:3' \
	               '->r|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->r|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->d|p:.5'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def baseline2(opts, input_shape, nb_classes, getstring_flag=False):
	model_string = 'cr|f:32,r:5' \
	               '->cr|f:64,r:5' \
	               '->mp|s:2,r:3' \
	               '->cr|f:64,r:3' \
	               '->cr|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->cr|f:128,r:5' \
	               '->ap|s:2,r:3' \
	               '->cr|f:64,r:4' \
	               '->ap|s:2,r:3' \
	               '->d|p:.5'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)

def springberg_baseline(opts, input_shape, nb_classes, getstring_flag=False):

	model_string = 'd|p:.25'\
					'->r|f:96,r:3,b:0' \
	               '->r|f:96,r:3,b:0' \
	               '->r|f:96,r:3,b:0' \
					'->mp|r:3'\
					'->d|p:.5'\
	               '->r|f:192,r:3,b:0' \
	               '->r|f:192,r:3,b:0' \
	               '->r|f:192,r:3,b:0' \
	               '->mp|r:3' \
	               '->d|p:.5'\
	               '->r|f:192,r:3,b:0' \
	               '->r|f:192,r:1,b:0' \
	               '->conv|f:'+str(nb_classes)+',r:1,b:0' \
	               '->apd|r:6'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def springberg_baseline2(opts, input_shape, nb_classes, getstring_flag=False):

	model_string = 'r|f:96,r:3,b:0' \
	               '->r|f:96,r:3,s:2,b:1' \
	               '->r|f:192,r:3,b:0' \
	               '->r|f:192,r:3,s:2,b:1' \
	               '->r|f:'+str(nb_classes)+',r:1,b:0' \
	               '->apd|r:6'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def nin_baseline(opts, input_shape, nb_classes, getstring_flag=False):

	model_string ='r|f:192,r:5,b:0' \
	               '->r|f:160,r:1,b:0' \
	               '->r|f:96,r:1,b:0' \
					'->mp|r:3'\
					'->d|p:.5'\
	               '->r|f:192,r:5,b:0' \
	               '->r|f:192,r:1,b:0' \
	               '->r|f:192,r:1,b:0' \
	               '->ap|r:3' \
	               '->d|p:.5'\
	               '->r|f:192,r:3,b:0' \
	               '->r|f:192,r:1,b:0' \
	               '->conv|f:'+str(nb_classes)+',r:1,b:0' \
	               '->apd|r:7'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def nin_besh2(opts, input_shape, nb_classes, getstring_flag=False):

	model_string ='besh|f:192,r:5,b:0,p:1' \
	               '->besh|f:160,r:1,b:0,p:1' \
	               '->besh|f:96,r:1,b:0,p:1' \
				'->leaffully|n:2'\
					'->mp|r:3'\
					'->d|p:.5'\
	               '->besh|f:192,r:5,b:0,p:1' \
	               '->besh|f:192,r:1,b:0,p:1' \
	               '->besh|f:192,r:1,b:0,p:1' \
	              '->leaffully|n:4'\
	              '->ap|r:3'\
	               '->d|p:.5'\
	               '->besh|f:192,r:3,b:0,p:1' \
	               '->besh|f:192,r:1,b:0,p:1' \
	               '->cshfixedfilter|f:'+str(nb_classes)+',r:1,b:0,p:1' \
	                                          '->globalpooling|r:7'
	nb_filter_list = [32, 32, 64, 128, 64, 64]
	filter_size_list = [5, 5, 3, 5, 3, 5, 3, 4, 3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts, input_shape, nb_classes, model_string=model_string, nb_filter_list=nb_filter_list,
	                 conv_filter_size_list=filter_size_list)
def get_model_from_db(identifier,opts,input_shape,nb_classes):
	model_fun = get_from_module(identifier, globals(), 'activation function')
	return model_fun(opts,input_shape,nb_classes)
def get_model_string_from_db(identifier,opts,input_shape,nb_classes):
	model_fun = get_from_module(identifier, globals(), 'activation function')
	return model_fun(opts, input_shape, nb_classes,getstring_flag=True)
if __name__ == '__main__':
	opts = default_opt_creator()
	functions = globals().copy()
	func_to_test = ['baseline2','baseline2_besh12','besh_crelu_12']
	for function in functions:
		if function not in func_to_test:
			continue
		model = functions.get(function)
		print function
		opts = opt_utils.set_model_string(opts,function)
		opts = opt_utils.set_dataset(opts,'cifar100')
		opt_utils.set_default_opts_based_on_model_dataset(opts)
		model = model(opts,(3,32,32),10)
		model.summary()
	for function in functions:
		if not function in func_to_test:
			continue
		model = functions.get(function)
		print function
		opts = opt_utils.set_model_string(opts,function)
		opts = opt_utils.set_dataset(opts,'cifar100')
		opt_utils.set_default_opts_based_on_model_dataset(opts)
		model = model(opts,(3,32,32),10)
		print model.count_params()

