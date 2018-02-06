import numpy as np


def lr_sched_function_load(dataset, network):
	return get('{}_{}'.format(dataset, network))


def cifar10_nin(index):
	lr = 80*[1e-1] + 80*[2e-2] + 80*[4e-3] + 80 * [8e-4] + 10 * [4e-3] + 100 * [4e-4]
	# lr = [2e-3] + [1e-2] + [2e-2] + 80 * [4e-2] + 10 * [4e-3] + 100 * [4e-4]
	print(('LearningRate:', lr[index]))
	return lr[index]


def cifar10_vgg(index):

	lr_val = .1*(.5**(index//25))
	return lr_val


def cifar100_vgg(index):
	lr_val = .1 * (.5 ** (index // 25))
	return lr_val
def default_LR_scheduler_funciton(index):
	lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-4]
	k = (index / 50)
	i = np.floor(k)
	if i < 0:
		i = 0
	if i >= lr.__len__():
		i = lr.__len__() - 1
	lr_s = lr[int(i)]
	print(('LearningRate:', float(lr_s)))
	return lr_s


def cifar100_nin(index):
	lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-4]
	k = (index / 50)
	i = np.floor(k)
	if i < 0:
		i = 0
	if i >= lr.__len__():
		i = lr.__len__() - 1
	lr_s = lr[int(i)]
	print(('LearningRate:', lr_s))
	return lr_s


def get(identifier):
	return globals()[identifier]
