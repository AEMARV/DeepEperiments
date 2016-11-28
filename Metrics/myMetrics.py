from keras import backend as K
import theano.tensor as T
from theano.printing import Print
def precision_recall_metric(y_true, y_pred):
	delta_pred_flag = True
	y_pred_max = K.max(y_pred, axis=1, keepdims=True)
	y_pred_delta_dist = K.equal(y_pred_max, y_pred)
	y_pred_total_per_class = K.sum(y_pred_delta_dist, axis=0, keepdims=False)
	if delta_pred_flag:
		y_pred_true_positive_matrix =T.mul(y_pred_delta_dist,y_true)   # y_pred matrix masked
	else:
		y_pred_true_positive_matrix = y_pred * y_true  # y_pred matrix masked
	y_pred_true_positive_total_vector = K.sum(y_pred_true_positive_matrix, axis=0, keepdims=False)
	y_true_total_per_class = K.sum(y_true, axis=0, keepdims=False)
	precision = y_pred_true_positive_total_vector / y_pred_total_per_class
	avg_precision = K.mean(precision)
	recall = y_pred_true_positive_total_vector / y_true_total_per_class
	avg_recall = K.mean(recall)
	return {'avg precision': avg_precision,'avg recall':avg_recall}

def test_pred_steps(y_true,y_pred):
	y_pred_max = K.max(y_pred, axis=1, keepdims=True)
	# y_pred_delta_dist = K.cast_to_floatx(K.equal(y_pred_max, y_pred))
	y_pred_delta_dist = K.equal(y_pred_max, y_pred)
	y_pred_is_boolean_test = K.all(K.equal(y_pred_delta_dist,0)+K.equal(y_pred_delta_dist,1))
	y_true_is_boolean_test = K.all(K.equal(y_true,0)+K.equal(y_true,1))
	masked_pred = y_pred_delta_dist*y_true
	masked_pred_is_boolean = K.all(K.equal(masked_pred,0)+K.equal(masked_pred,1))
	sum_pred_row_sum = K.sum(masked_pred,axis=1,keepdims=False)
	sum_pred = Print("result:")(K.sum(masked_pred))
	sum_pred_row_boolean_test = K.all(K.equal(sum_pred_row_sum,1)+K.equal(sum_pred_row_sum,0))
	return {"sumpred":sum_pred}