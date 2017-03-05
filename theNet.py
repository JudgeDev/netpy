import numpy as np

from mnist import Mnist
from sgd_optimization import SgdOptimization
from net import NetPy
from torontoNet import TorontoNet


if __name__ == '__main__':
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
			
	#print('train inputs: {}'.format(X.shape))
	#print('train labels: {}'.format(y.shape))
	
	n_train_batches = X.shape[0] // batch_size
	n_valid_batches = valid_set_x.shape[0] // batch_size
	n_test_batches = test_set_x.shape[0] // batch_size
	
	#print('number of minibatches of size {}: {}, {}, {}'.format(
		#batch_size, n_train_batches, n_valid_batches, n_test_batches))
	
	# number of training cases
	m = X.shape[0]
	#print('number of training cases: {}'.format(m))
	
	# randomly select 100 data points to display
	# random permutation of training cases
	sel = np.random.permutation(m)
	# select 100
	sel = sel[:100]	
	#dataset.displayData(X[sel, :])	
		
	# instantiate TorontoNet
	# constructor needs:
	# net parameters: # hidden units
	tNet = TorontoNet(25)
	#tNet.test()
	# optimize net
	# needs:
	# datasets
	# learning parameters: # iterations, minibatch size,
	# 	learning rate, momentum multiplier
	# regularization: weight decay coefficient, early stopping flag
	tNet.optimization(datasets, 100, 600, 0.1, 0.9, 0, False)
	


	
	
	
	# instantiate Optimization
	#opt = SgdOptimization(datasets)
	#opt.train()

	#net = pyNet1()
	
