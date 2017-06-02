
			# elif component == 'max_entropy_instance':
			# 	#TODO: COMPLETE THIS
			# 	x = node_list_to_list(x)
			#
			# elif component == 'fin':
			# 	x = node_list_to_list(x)
			# 	if not x.__len__()==1:
			# 		raise ValueError('output node is a list of tensor, Probably forgot about merging branch')
			# 	x = x[0]
			# 	return Model(input=img_input, output=x)
			# #
			# elif component =='besh':
			# 	#binary expand shared
			# 	nb_filter = param['f']
			# 	f_size = int(param['r'])
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	stride=1
			# 	conv_layer_to_pass = Conv2D(int(nb_filter*expand_rate), (f_size, f_size), activation=None,
			# 	                                   input_shape=input_shape, padding='same', kernel_regularizer=w_reg,
			# 	                                  name='CONV_L'+str(layer_index_t))
			# 	x = conv_birelu_expand_on_list_shared(input_tensor_list=x,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,batch_norm=0,
			# 	                               leak_rate=0,child_p = 0,conv_layer=conv_layer_to_pass,drop_path_rate
			# 	                                      = dropout)
			# elif component =='beshpermute':
			# 	#the output tensors would be 2^(n+C) if we have n input tensors and C as kernel channels
			# 	nb_filter = param['f']
			# 	f_size = int(param['r'])
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	if param.has_key('rand'):
			# 		random_permute_flag = int(param['rand'])
			# 	max_perm = 2
			# 	if param.has_key('max_perm'):
			# 		max_perm = int(param['max_perm'])
			#
			# 	conv_layer_to_pass = Conv2D(int(nb_filter*expand_rate), (f_size, f_size), activation=None,
			# 	                                   input_shape=input_shape, padding='same', kernel_regularizer=w_reg,
			# 	                                  name='CONV_L'+str(layer_index_t))
			# 	x = conv_birelu_expand_on_list_shared_permute_channels(input_tensor_list=x,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,batch_norm=0,
			# 	                               leak_rate=0,child_p = 0,conv_layer=conv_layer_to_pass,drop_path_rate
			# 	                                      = dropout,max_perm=max_perm,random_permute_flag=random_permute_flag)
			# elif component =='shdense':
			# 	max = 0
			# 	if param.has_key('m'):
			# 		max = float(param['m'])
			# 	n = int(param['n'])
			# 	if n==-1:
			# 		n = nb_classes
			# 	d = param['do']
			# 	x = node_list_to_list(x)
			# 	flatten_flag = True
			# 	dense = Dense(n)
			# 	res  = []
			# 	for tensor in x:
			# 		tensor_f = Flatten()(tensor)
			# 		if not d ==0:
			# 			tensor_f = Dropout(d)(tensor_f)
			# 		tensor_f = dense(tensor_f)
			# 		res+=[tensor_f]
			# 	if res.__len__()==1:
			# 		res_sum = res[0]
			# 	else:
			# 		res_sum = add(res)
			# 	if max==0:
			# 		x= res_sum
			# 	else:
			# 		res_max = MaxoutDenseOverParallel()(res)
			# 		x = res_max
			# 		if max ==.5:
			# 			x = add([res_max,res_sum])
			# 	x = Activation('softmax')(x)
			# elif component =='shdense_legacy':
			# 	max = 0
			# 	if param.has_key('m'):
			# 		max = float(param['m'])
			# 	n = int(param['n'])
			# 	if n==-1:
			# 		n = nb_classes
			# 	d = param['do']
			# 	x = node_list_to_list(x)
			# 	flatten_flag = True
			# 	dense = Dense(n)
			# 	res  = []
			# 	for tensor in x:
			# 		tensor_f = Flatten()(tensor)
			# 		if not d ==0:
			# 			tensor_f = Dropout(d)(tensor_f)
			# 		tensor_f = dense(tensor_f)
			# 		res+=[tensor_f]
			#
			# 	res_sum = add(res)
			# 	if max==0:
			# 		x= res_sum
			# 	else:
			# 		# res = K.expand_dims()
			# 		res_max = MaxoutDenseOverParallel()(res)
			# 		x = res_max
			# 		if max ==.5:
			# 			x = add([res_max,res_sum])
			# elif component == 'shdensedoi':
			# 	# dropout instanse
			# 	# tdo is tensor dropout which is the usuall dropout
			# 	# ido is instance dropout
			# 	max = 0
			# 	if param.has_key('m'):
			# 		max = float(param['m'])
			# 	n = int(param['n'])
			# 	if n == -1:
			# 		n = nb_classes
			# 	dense_dropout = float(param['tdo'])
			# 	classification_dropout = float(param['ido'])
			# 	x = node_list_to_list(x)
			# 	if classification_dropout ==-1:
			# 		classification_dropout = .75
			# 	flatten_flag = True
			# 	dense = Dense(n)
			# 	res = []
			# 	for tensor in x:
			# 		tensor_f = Flatten()(tensor)
			# 		if not dense_dropout == 0:
			# 			tensor_f = Dropout(dense_dropout)(tensor_f)
			# 		tensor_f = dense(tensor_f)
			# 		tensor_f = InstanceDropout(classification_dropout)(tensor_f)
			# 		res += [tensor_f]
			# 	res_sum = add(res)
			# 	if max == 0:
			# 		x = res_sum
			# 	else:
			# 		# res = K.expand_dims()
			# 		res_max = MaxoutDenseOverParallel()(res)
			# 		x = add([res_max, res_sum])
			# elif component == 'mp':
			# 	f_size = param['r']
			# 	strides = int(2)
			# 	x = max_pool_on_list(input_tensor_list=x, strides=(strides, strides), layer_index=layer_index_t,
			# 	                     pool_size=f_size)
			# 	conv_nb_filterindex-=1
			# elif component=='ap':
			# 	f_size = int(param['r'])
			# 	x =avg_pool_on_list(input_tensor_list=x,strides=(2,2),layer_index=layer_index_t,
			# 	                    pool_size=f_size)
			# 	conv_nb_filterindex-=1
			# #TODO Create a general dropout later. its not complete yet.remove todo after complete
			# elif component=='gdropout':
			# 	# general dropout layer including instance dropout and 2d dropout
			# 	dense_dropout = float(param['tdo'])
			# 	classification_dropout = float(param['ido'])
			# 	for tensor in x:
			# 		tensor_f = Flatten()(tensor)
			# 		if not dense_dropout == 0:
			# 			tensor_f = Dropout(dense_dropout)(tensor_f)
			# 		tensor_f = dense(tensor_f)
			# 		tensor_f = InstanceDropout(classification_dropout)(tensor_f)
			# 		res += [tensor_f]
			# 	res_sum = add(res)
			# elif component=='leaffully':
			# 	if param.has_key('n'):
			# 		n = param['n']
			# 	else:
			# 		n = 0
			# 	x = node_list_to_list(x)
			# 	if param.has_key('chw'):
			# 		if param['chw']==1:
			# 			x = FullyConnectedTensors(int(n),shared_axes=[2,3])(x)
			# 	else:
			# 		x = FullyConnectedTensors(int(n))(x)
			# elif component=='cr':
			# 	nb_filter = param['f']
			# 	f_size = int(param['r'])
			# 	conv_layer_to_pass = Conv2D(int(nb_filter * expand_rate), (f_size, f_size), activation=None,
			# 	                                   input_shape=input_shape, padding='same', kernel_regularizer=w_reg,
			# 	                                    name='CONV_L'+str(layer_index_t))(x[0])
			# 	tensors = Birelu('relu',name='BER_CR'+str(layer_index_t))(conv_layer_to_pass)
			# 	x = [concatenate(tensors,axis=1)]
			# elif component=='e':
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			# 	x = conv_birelu_expand_on_list(input_tensor_list=x,nb_filter=int(nb_filter * expand_rate/branch),
			# 	                               filter_size=f_size,
		     #                           input_shape=input_shape, w_reg=w_reg,
		     #                   gate_activation=get_gate_activation(opts), layer_index=layer_index_t,border_mode='same')
			# 	branch=2*branch
			# elif component =='rsh':
			# 	#binary expand shared
			# 	nb_filter = int(param['f'])
			# 	f_size = int(param['r'])
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			#
			# 	conv_layer_to_pass = Conv2D(int(nb_filter*expand_rate), (f_size, f_size), activation=None,
			# 	                                   input_shape=input_shape, padding='same', kernel_regularizer=w_reg,
			# 	                                  kernel_initializer='he_normal')
			# 	x = conv_relu_expand_on_list_shared(input_tensor_list=x,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,batch_norm=0,
			# 	                               leak_rate=0,child_p = 0,conv_layer=conv_layer_to_pass,drop_path_rate
			# 	                                      = dropout)
			# elif component == 'cshfixedfilter':
			# 	#expand rate does not affect num of filter for this convolution
			# 	nb_filter = int(param['f'])
			# 	f_size = int(param['r'])
			# 	conv_to_pass = Conv2D(int(nb_filter), (f_size, f_size),
			# 	 activation = None, input_shape = input_shape, padding = 'same', W_regularizer = w_reg,
			# 	                      kernel_initializer='he_normal'
			# 	)
			# 	x= node_list_to_list(x)
			# 	for i in range(x.__len__()):
			# 		x[i]= conv_to_pass(x[i])
			# if component == 'xaesh':
			# 	# binary expand shared
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	conv_layer_to_pass = Conv2D(int(nb_filter * expand_rate), f_size, f_size, activation=None,
			# 	                                   input_shape=input_shape, padding='same', kernel_regularizer=w_reg, )
			# 	x = conv_xavr_expand_on_list_shared(input_tensor_list=x, gate_activation=get_gate_activation(opts),
			# 	                                      layer_index=layer_index_t, batch_norm=0, leak_rate=0, child_p=0,
			# 	                                      conv_layer=conv_layer_to_pass, drop_path_rate=dropout)
			# if component == 'xaresh':
			# 	# binary expand shared
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	conv_layer_to_pass = Conv2D(int(nb_filter * expand_rate), f_size, f_size, activation=None,
			# 	                                   input_shape=input_shape, padding='same', kernel_regularizer=w_reg, )
			# 	x = conv_xavrrelu_expand_on_list_shared(input_tensor_list=x, gate_activation=get_gate_activation(opts),
			# 	                                      layer_index=layer_index_t, batch_norm=0, leak_rate=0, child_p=0,
			# 	                                      conv_layer=conv_layer_to_pass, drop_path_rate=dropout)
			# if component == 'rbe':
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	if param.has_key('leak'):
			# 		leak_rate = param['leak']
			# 	if param.has_key('bn'):
			# 		batch_norm = param['bn']
			# 	if param.has_key('cp'):
			# 		child_probability = param['cp']
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			#
			# 	x = conv_birelu_expand_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                               filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same',relu_birelu_switch=dropout,batch_norm=batch_norm,
			# 	                               leak_rate=leak_rate,child_p = child_probability)
			#
			# 	branch = 2 * branch
			# if component == 'pre':
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	if param.has_key('bn'):
			# 		batch_norm = param['bn']
			# 	if param.has_key('cp'):
			# 		child_probability = param['cp']
			# 	if param.has_key('counter'):
			# 		if param['counter']==1:
			# 			counter=True
			# 		else:
			# 			counter=False
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			#
			# 	x = conv_prelu_expand_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                               filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same', relu_birelu_switch=dropout, batch_norm=batch_norm,
			# 	                               leak_rate=0, child_p=child_probability,prelu_counter=counter)
			#
			# 	branch = 2 * branch
			# if component == 'rbeg':
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	if param.has_key('leak'):
			# 		leak_rate = param['leak']
			# 	if param.has_key('bn'):
			# 		batch_norm = param['bn']
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			#
			# 	x = conv_birelu_expand_on_list_general_leak(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                               filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same',relu_birelu_switch=dropout,batch_norm=batch_norm,
			# 	                               leak_rate=leak_rate)
			# 	branch = 2 * branch
			# if component == 'rben':
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			# 	x = conv_birelunary_expand_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                               filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same',relu_birelu_switch=dropout)
			# 	branch = (2**nb_filter) * branch
			# if component=='s':
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			# 	x = conv_birelu_swap_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/(branch/2)),
			# 	                             filter_size=f_size,
			# 	                               input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same')
			# if component=='rbs':
			# 	if param.has_key('p'):
			# 		dropout = param['p']
			# 	nb_filter = param['f']
			# 	f_size = param['r']
			# 	x = conv_birelu_swap_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			# 	                             filter_size=f_size,
			# 	                               input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same',relu_birelu_switch=dropout)
			# if component=='rm':
			# 	nb_filter = param['f']
			# 	branch = branch / 2
			# 	f_size = param['r']
			# 	x = conv_relu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			# 	                              filter_size=f_size,
			# 	                               input_shape=input_shape, w_reg=w_reg,
			# 	                               gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                               border_mode='same')
			# if component=='am':
			# 	nb_filter = param['f']
			# 	branch = branch / 2
			# 	f_size = param['r']
			# 	x = conv_relu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate/branch),
			# 	                              filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                              gate_activation='avr', layer_index=layer_index_t, border_mode='same')
			# if component == 'bm':
			# 	nb_filter = param['f']
			# 	branch = branch / 2
			# 	f_size = param['r']
			# 	x = conv_birelu_merge_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                              filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                              gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                              border_mode='same')
			#
			# if component == 'apd':
			# 	f_size = param['r']
			# 	x = avg_pool_on_list(input_tensor_list=x, strides=(1, 1), layer_index=layer_index_t, pool_size=f_size)
			# 	conv_nb_filterindex -= 1
			# 	no_class_dense = True
			#
			# if component == 'conv':
			# 	f_size = param['r']
			# 	nb_filter = param['f']
			# 	stride = 1
			# 	if param.has_key('s'):
			# 		stride = param['s']
			# 	border_mode = 'same'
			# 	if param.has_key('b'):
			# 		border_mode = param['b']
			# 		if border_mode == 1:
			# 			border_mode = 'valid'
			# 		else:
			# 			border_mode = 'same'
			#
			# 	x = conv_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                      filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                      gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                      border_mode=border_mode, stride=stride, b_reg=b_reg)
			# if component == 'r':
			# 	f_size = int(param['r'])
			# 	nb_filter = int(param['f'])
			# 	stride=1
			# 	if param.has_key('s'):
			# 		stride = int(param['s'])
			# 	border_mode = 'same'
			# 	if param.has_key('b'):
			# 		border_mode = param['b']
			# 		if border_mode ==1:
			# 			border_mode= 'valid'
			# 		else:
			# 			border_mode= 'same'
			#
			#
			# 	x = conv_relu_on_list(input_tensor_list=x, nb_filter=int(nb_filter * expand_rate / branch),
			# 	                              filter_size=f_size, input_shape=input_shape, w_reg=w_reg,
			# 	                              gate_activation=get_gate_activation(opts), layer_index=layer_index_t,
			# 	                              border_mode=border_mode,stride=stride,b_reg= b_reg)
			# if component=='d':
			# 	p = param['p']
			# 	x = dropout_on_list(input_tensor_list=x,p=p,layer_index=layer_index_t)
			# if component=='c':
			# 	n = param['n']
			# 	branch = branch / 2
			# 	if n == -1:
			# 		while type(x[0])is list:
			# 			x = concat_on_list(input_tensor_list=x, n=1, layer_index=layer_index_t)
			# 	else:
			# 		for i in range(int(n)):
			# 			x = concat_on_list(input_tensor_list=x,n=n,layer_index=layer_index_t)
			#
			# if component =='flattensh_legacy':
			# 	res = []
			# 	x = node_list_to_list(x)
			# 	flatten_flag = True
			# 	for tensor in x:
			# 		tensor_f = Flatten()(tensor)
			# 		res+=[tensor_f]
			# 		x = res
			# if component =='dropoutsh':
			# 	#TODO: remove flatten and dense from this component
			# 	d = param['p']
			# 	x = node_list_to_list(x)
			# 	flatten_flag = True
			# 	res  = []
			# 	for tensor in x:
			# 		tensor_f = Dropout(d)(tensor)
			# 		res+=[tensor_f]
			# 	x = res
			# if component =='densesh_legacy':
			# 	#TODO:  remove dropout and flatten from densesh .
			# 	n = int(param['n'])
			# 	if param.has_key('act'):
			# 		activation = param['act']
			# 	else:
			# 		activation = None
			# 	if n==-1:
			# 		n = nb_classes
			# 		no_class_dense==True
			# 	x = node_list_to_list(x)
			# 	dense = Dense(n,activation=activation,kernel_initializer='he_normal')
			# 	res  = []
			# 	for tensor in x:
			# 		tensor = dense(tensor)
			# 		res+=[tensor]
			# 	x= res
			# if component == 'merge':
			# 	# TODO:  remove dropout and flatten from densesh .
			# 	if n == -1:
			# 		n = nb_classes
			# 	mode = param['mode']
			# 	x = node_list_to_list(x)
			# 	x = merge(x,mode=mode)
			# if component == 'globalpooling':
			# 	max = 0
			# 	if param.has_key('m'):
			# 		max = float(param['m'])
			# 	f_size = param['r']
			# 	res = node_list_to_list(x)
			# 	if res.__len__()==1:
			# 		res_sum=res[0]
			# 	else:
			# 		res_sum = add(res)
			# 	x = avg_pool_on_list(input_tensor_list=[res_sum], strides=(1, 1), layer_index=layer_index_t, \
			# 	                                                                                 pool_size=f_size)
			# 	no_class_dense = True
			# 	flatten_flag = False
			#
			# if component == 'shdense3':
			# 	max = 0
			# 	if param.has_key('m'):
			# 		max = float(param['m'])
			# 	n = int(param['n'])
			# 	if n == -1:
			# 		n = nb_classes
			# 	dense_dropout = float(param['dode'])
			# 	classification_dropout = float(param['doclas'])
			# 	x = node_list_to_list(x)
			# 	flatten_flag = True
			# 	dense = Dense(n)
			# 	res = []
			# 	for tensor in x:
			# 		tensor_f = Flatten()(tensor)
			# 		if not dense_dropout == 0:
			# 			tensor_f = Dropout(dense_dropout)(tensor_f)
			# 		tensor_f = dense(tensor_f)
			# 		tensor_f = InstanceDropout(classification_dropout)(tensor_f)
			# 		res += [tensor_f]
			# 	res_sum = add(res)
			# 	if max == 0:
			# 		x = res_sum
			# 	else:
			# 		# res = K.expand_dims()
			# 		res_max = MaxoutDenseOverParallel()(res)
			# 		x = res_max
			# 		if max == .5:
			# 			x = add([res_max, res_sum])
			# 		x = add([res_max, res_sum])
