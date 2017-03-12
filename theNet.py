import sys
print(sys.path)
import os
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

from mnist import Mnist
from net import NetPy
from torontoNet import TorontoNet
from sgd_optimization import SgdOptimization
from mlp_optimization import MlpOptimization

if __name__ == '__main__':

	"""
	# evdev - not detecting any devices
	import evdev

		
		# checking evdev
		print('checking devices...')
		devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
		for device in devices:
			print(device.fn, device.name, device.phys)
	"""	
	print('Running theNet as main')
	
	batch_size = 600
	
	# instanciate mnist dataset
	dataset = Mnist()
	# load date from the path of the MNIST dataset file from:
	#	http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
	dataset.load_data('mnist.pkl.gz')
	datasets = dataset.get_datasets()

	X, y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]
	
	# convert from MNIST to TorontoNet
	# dict of dicts of matrices_m,inputs/outputs
	# convert y_m,1 labels into yv_m,10 vectors using lookup
	lookup = np.eye(10)
	datas = defaultdict(dict)    
	datas['training']['inputs'] = datasets[0][0].T
	datas['training']['targets'] = lookup[datasets[0][1], :].T
	datas['validation']['inputs'] = datasets[1][0].T
	datas['validation']['targets'] = lookup[datasets[1][1], :].T
	datas['test']['inputs'] = datasets[2][0].T
	datas['test']['targets'] = lookup[datasets[2][1], :].T

	n_train_batches = X.shape[0] // batch_size
	n_valid_batches = valid_set_x.shape[0] // batch_size
	n_test_batches = test_set_x.shape[0] // batch_size
	
	# number of training cases
	m = X.shape[0]
	
	# randomly select 100 data points to display
	# random permutation of training cases
	sel = np.random.permutation(m)
	# select 100
	sel = sel[:100]	
	#dataset.displayData(X[sel, :])
	
	"""
	# USPS data
	# data is dict
	data = loadmat(os.path.join(os.getcwd(), 'data/data.mat'))
	data_sets = data['data'][0][0]
	# Convert from MNIST to TorontoNet
	# dict of dicts of matrices_m,inputs/outputs
	datas = defaultdict(dict)    
	datas['training']['inputs'] = data_sets['training']['inputs'][0][0]
	datas['training']['targets'] = data_sets['training']['targets'][0][0]
	datas['validation']['inputs'] = data_sets['validation']['inputs'][0][0]
	datas['validation']['targets'] = data_sets['validation']['targets'][0][0]
	datas['test']['inputs'] = data_sets['test']['inputs'][0][0]
	datas['test']['targets'] = data_sets['test']['targets'][0][0]
	
	"""


	# TorontoNet test
	# instantiate TorontoNet
	# constructor needs:
	# net parameters: #input units, #hidden units, #output units
	#tNet = TorontoNet(784, 37, 10)
	# optimize net, needs: datasets,
	# learning parameters: # iterations, minibatch size,
	# 	learning rate, momentum multiplier
	# regularization: weight decay coefficient, early stopping flag
	#tNet.optimization(datas, 1000, 100, 0.35, 0.9, 0, True)
	
	# Logistic regression test
	# instantiate sgd Optimization
	#opt = SgdOptimization(datasets)
	#opt.build()
	#opt.train()

	# Multilayer Perceptron test
	# instantiate mpl Optimization
	# constructor needs datasets, and has default values of:
	# learning_rate=0.01, L1_reg=0.00, L2_reg=0.001, n_epochs=1000,
	# batch_size=1000, n_hidden=500
	
	mlp = MlpOptimization(datasets)
	mlp.build()
	mlp.train()

	
