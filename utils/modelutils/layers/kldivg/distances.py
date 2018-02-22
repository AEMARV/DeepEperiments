import keras.backend as K
def kl_v1(cross_xprob_kerlog, cross_xlog_kerprob, ent_x, ent_ker):
	distance = 0
	distance += cross_xlog_kerprob
	distance += cross_xprob_kerlog
	distance += ent_x
	distance += ent_ker
	return distance
def kl_v2(cross_xprob_kerlog, cross_xlog_kerprob, ent_x, ent_ker):
	distance = 0
	distance += cross_xlog_kerprob
	distance += cross_xprob_kerlog
	distance += ent_x
	#distance += ent_ker
	return distance
def kl_v3(cross_xprob_kerlog, cross_xlog_kerprob, ent_x, ent_ker):
	distance = 0
	distance += cross_xlog_kerprob
	#distance += cross_xprob_kerlog
	#distance -= ent_x
	distance += ent_ker
	return distance
def kl_loss_label_centric(y_true,y_pred):
	loss = 0
	y_true_cut = K.clip(y_true, K.epsilon(), 1-K.epsilon())
	y_true_cut = y_true_cut/K.sum(y_true_cut,axis=[-1],keepdims=True)
	ent_preds = -K.exp(y_pred)*y_pred
	ent_preds = K.sum(ent_preds, axis=[-1])
	ent_labels = -K.log(y_true_cut)*y_true_cut
	ent_labels = K.sum(ent_labels,axis=[-1])
	cross_xlog_mp = y_true * y_pred
	cross_xlog_mp = K.sum(cross_xlog_mp,axis=[-1])
	cross_xp_mlog = K.exp(y_pred)*K.log(y_true_cut)
	cross_xp_mlog = K.sum(cross_xp_mlog, axis=[-1])

	loss += cross_xp_mlog
	loss += ent_preds
	return -loss
def kl_loss_model_centric(y_true,y_pred):
	loss = 0
	y_true_cut = K.clip(y_true, K.epsilon(), 1-K.epsilon())
	y_true_cut = y_true_cut/K.sum(y_true_cut,axis=[-1],keepdims=True)
	ent_preds = -K.exp(y_pred)*y_pred
	ent_preds = K.sum(ent_preds, axis=[-1])
	ent_labels = -K.log(y_true_cut)*y_true_cut
	ent_labels = K.sum(ent_labels, axis=[-1])
	cross_xlog_mp = y_true * y_pred
	cross_xlog_mp = K.sum(cross_xlog_mp, axis=[-1])
	cross_xp_mlog = K.exp(y_pred)*K.log(y_true_cut)
	cross_xp_mlog = K.sum(cross_xp_mlog, axis=[-1])

	loss += cross_xlog_mp
	loss += ent_labels
	return -loss
def kl_loss_both_centric(y_true,y_pred):
	loss = 0
	y_true_cut = K.clip(y_true, K.epsilon(), 1-K.epsilon())
	y_true_cut = y_true_cut/K.sum(y_true_cut, axis=[-1], keepdims=True)
	ent_preds = -K.exp(y_pred)*y_pred
	ent_preds = K.sum(ent_preds, axis=[-1])
	ent_labels = -K.log(y_true_cut)*y_true_cut
	ent_labels = K.sum(ent_labels, axis=[-1])
	cross_xlog_mp = y_true * y_pred
	cross_xlog_mp = K.sum(cross_xlog_mp, axis=[-1])
	cross_xp_mlog = K.exp(y_pred)*K.log(y_true_cut)
	cross_xp_mlog = K.sum(cross_xp_mlog, axis=[-1])

	loss += cross_xp_mlog
	loss += ent_preds
	loss += cross_xlog_mp
	loss += ent_labels
	return -loss
