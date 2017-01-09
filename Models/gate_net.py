import numpy as np
from keras.layers import Input
from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from keras.models import Model
from Layers.gate_layer import gate_layer,gated_layers_sequence
def gated_net_functional_wrapped(nb_filter = 4,filter_size=3,depth =5 , input_shape=(3,None,None),input_tensor=None):
	if input_tensor==None:
		img_input = Input(shape=input_shape, name='image_batch')
	else:
		img_input = input_tensor

	out2 = gated_layers_sequence(input_tensor=img_input,total_layers=depth,nb_filter=nb_filter,filter_size=filter_size,
	                             input_shape=input_shape)
	model = Model(input=img_input,output=out2)
	return model
if __name__ == '__main__':
    model = gated_net_functional_wrapped()
    model.summary()
    plot(model, to_file='model.png')
    sgd = SGD(lr=0.01, momentum=.9, decay=5 * 1e-4, nesterov=True)
    model.compile(optimizer=sgd,loss = 'mse')
    np.random.seed(0)
    input_im = np.random.randint(0,256,(1,3,4,4))
    print "input data: \n",input_im
    a = model.predict(input_im)
    print "Predictions"
    print a
    print a.shape

