import keras.backend as K
from keras.regularizers import Regularizer
class EntropyRegularizer(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """
    def get_config(self):
        return {'l1': float(self.l1)}

    def __init__(self, coef):
        self.coef = K.cast_to_floatx(coef)

    def __call__(self, x):
        xnorm = x - K.logsumexp(x, axis=2, keepdims=True)
        ent = -xnorm * K.exp(xnorm)
        ent = self.coef * K.sum(ent)
        return -ent




