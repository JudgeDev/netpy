import sys
#print(sys.path)
if 'Cuda' in sys.path: print('Cuda in path')
import os
#os.environ['THEANO_FLAGS']= 'mode=FAST_RUN,device=gpu,floatX=float32'
from collections import defaultdict

import numpy as np
from scipy.io import loadmat
import theano
import theano.tensor as T

from mnist import Mnist

class NetClass:
	"""Neural net class

	Initially, the net will be limited to a  multilayer perceptron.
	This is a feedforward artificial neural network model
	that has one layer or more of hidden units and nonlinear activations.
	Intermediate layers usually have as activation function tanh or the
	sigmoid function (defined here by a ‘‘HiddenLayer‘‘ class).
	The output layer is a softmax layer (defined here by a ‘‘OutputLayer‘‘
	class).
	"""
	
	def __init__(self, layers):
		
		"""Initialize the parameters for the multilayer perceptron
		
		ARGS:
			layers (list): list of tuples
				# of units(int), ??
		"""
		print('Initialising {} layer network, with {} units'.format(
			len(layers), [layer[0] for layer in layers]))
		
		
		X = 0  # input vector
		l = {}  # dictionary of layers
		for layer, data in enumerate(layers[:-1]):
			# create layer according to input parameters
			# and connected to previous layer
			l[layer+1] = Layer(
				rng=0,  # random number generator??
				# input is input data or ouput from previous layer
				x = X if layer == 0 else l[layer].y,
				i=layers[layer][0],  # # of inputs
				o=data[0],  # # of outputs
				#s=data[1]  # activation function
			)
			# create layer weights here on in layer class?	
			
		print('layers: {}'.format(l))
			
		"""
		# dict of weights W[layer]_out,in
		W, b = {}, {}			W[layer] = np.zeros((data[0], layers[layer-1][0]))
			b[layer] = np.zeros(data[0])
			# ouput
			if layer == len(layers):
				pass
		
		print([W[layer+1].shape for layer in range(len(layers)-1)])
		
		"""
		
		"""
		# Since we are dealing with a one hidden layer MLP, this will translate
		# into a HiddenLayer with a tanh activation function connected to the
		# LogisticRegression layer; the activation function can be replaced by
		# sigmoid or any other nonlinear function
		self.hiddenLayer = hl.HiddenLayer(
			rng=rng,
			input=input,
			n_in=n_in,
			n_out=n_hidden,
			activation=T.tanh
		)
		
		# The logistic regression layer gets as input the hidden units
		# of the hidden layer
		self.logRegressionLayer = lr.LogisticRegression(
			input=self.hiddenLayer.output,
			n_in=n_hidden,
			n_out=n_out
		)
		"""

class Layer:
	"""General purpose neural network layer class
	
	"""	
	def __init__(self, rng, x, i, o, activation=T.tanh):
		"""Generic layer of a neural net
		
		The units are fully-connected and have specified activation fnct.
		
		ARGS:
			rng(np.random.RandomState): a random number generator used
				to initialize weights
			input(T.dmatrix): a symbolic tensor of shape (n_examples, n_in)
			n_in (int): dimensionality of input
			n_out (int): number of hidden units (ie. output from this layer)
			activation(theano.Op or function): Non linearity to be applied
				in the hidden layer
		
		Weight and bias are theano shared variables.
		Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).
		
		??ADAPT
		# ‘W‘ is initialized with ‘W_values‘ which is uniformely sampled
		# from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
		# for tanh activation function
		# the output of uniform if converted using asarray to dtype
		# theano.config.floatX so that the code is runable on GPU
		# Note : optimal initialization of weights is dependent on the
		#        activation function used (among other things).
		#        For example, results presented in [Xavier10] suggest that you
		#        should use 4 times larger initial weights for sigmoid
		#        compared to tanh
		#        We have no info for other function, so we use the same as
		#        tanh.			
		"""
		self.i = i

		# initialise W and b
		# use passed value if given or else use shared?
		"""
		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
					W_values *= 4
		
			W = theano.shared(value=W_values, name='W', borrow=True)
		
		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)
		
		self.W = W
		self.b = b
		"""
		# temp values
		self.W = 0
		self.b = 0
		
		z = T.dot(i, self.W) + self.b
		self.y = (
			z if activation is None
			else activation(z)
		)
		# parameters of the model
		self.params = [self.W, self.b]		

if __name__ == '__main__':

	print('Running NetClass as main')
	
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
	m = datasets[0][0].shape[0]
	print('Number of training cases in dataset: {}'.format(m))
	
	# instantiate NetClass
	net = NetClass([(28*28,), (37,), (10,)])

