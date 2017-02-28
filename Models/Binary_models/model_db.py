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
	               '->s|f:256,r:4' \
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
	               '->ap|s:2,r:3' \

	nb_filter_list = [32,32,64,128,64,64]
	filter_size_list = [5,5,3,5,3,5,3,4,3]
	if getstring_flag:
		return {'string': model_string, 'nb_filter': nb_filter_list, 'filter_size': filter_size_list}
	return get_model(opts,input_shape,nb_classes,model_string=model_string,nb_filter_list=nb_filter_list,
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
	for function in functions:
		if not function[0] in ['b','l']:
			continue
		model = functions.get(function)
		print function
		opts = opt_utils.set_model_string(opts,function)
		opts = opt_utils.set_dataset(opts,'cifar100')
		opt_utils.set_default_opts_based_on_model_dataset(opts)
		model = model(opts,(3,32,32),10)
		model.summary()
	for function in functions:
		if not function[0] in ['b','l']:
			continue
		model = functions.get(function)
		print function
		opts = opt_utils.set_model_string(opts,function)
		opts = opt_utils.set_dataset(opts,'cifar100')
		opt_utils.set_default_opts_based_on_model_dataset(opts)
		model = model(opts,(3,32,32),10)
		print model.count_params()

