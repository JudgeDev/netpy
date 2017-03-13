import sys
#print(sys.path)
import os
from collections import defaultdict

import numpy as np
from scipy.io import loadmat

import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu0')

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
	
	# general parameters for comparison
	# network
	D = 28 * 28  # number of input dimensions for MNIST
	H = 37  # number of hidden units
	L = 10  # number of labels for digit classification
	# learning
	mb = 100   # mini batch size
	lda = 0.35  # learning rate lambda
	iters = 1000  # number of iterations
	
	
	# instanciate mnist dataset
	mnist = Mnist()
	# load date from the path of the MNIST dataset file from:
	#	http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
	mnist.load_data('mnist.pkl.gz')
	# get train, validation and test data sets
	# each set is a tuple(input, target):
	# input: numpy.ndarray - X_D,m of pixel values [0,1]
	# target: numpy.ndarray - y_m of labels [0 .. 9]
	datasets = mnist.get_datasets()

	print(datasets[0][0].shape)
	print(datasets[0][1].shape)
	m = datasets[0][0].shape[0]
	print('Number of training cases in dataset: {}'.format(m))
	
	# convert data for TorontoNet
	# convert to dictionaries
	# dict of dicts of matrices_m,inputs/outputs
	# convert y_m,1 labels into yv_m,10 vectors using lookup
	lookup = np.eye(10)
	t_datas = defaultdict(dict)    
	t_datas['training']['inputs'] = datasets[0][0].T
	t_datas['training']['targets'] = lookup[datasets[0][1], :].T
	t_datas['validation']['inputs'] = datasets[1][0].T
	t_datas['validation']['targets'] = lookup[datasets[1][1], :].T
	t_datas['test']['inputs'] = datasets[2][0].T
	t_datas['test']['targets'] = lookup[datasets[2][1], :].T

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
	tNet = TorontoNet(D, H, L)
	# optimize net, needs: datasets,
	# learning parameters: # iterations, minibatch size,
	# 	learning rate, momentum multiplier
	# regularization: weight decay coefficient, early stopping flag
	tNet.optimization(t_datas, iters, mb, lda, 0.9, 0, True)
	
	# Logistic regression test
	# instantiate sgd Optimization
	#opt = SgdOptimization(datasets, lda, iters, mb)
	#opt.build()
	#opt.train()

	# Multilayer Perceptron test
	# instantiate mpl Optimization
	# constructor needs datasets, and has default values of:
	# learning_rate=0.01, L1_reg=0.00, L2_reg=0.001, n_epochs=1000,
	# batch_size=1000, n_hidden=37

	mlp = MlpOptimization(datasets, lda, 0.00, 0.00, iters, mb, H)
	mlp.build()
	mlp.train()

	
