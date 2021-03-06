import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu0')

class HiddenLayer(object):
	
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
		activation=T.tanh):
		"""Typical hidden layer of a MLP
		
		The units are fully-connected and have sigmoidal activation fnct.
		Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).
		NOTE : The nonlinearity used here is tanh
		Hidden unit activation is given by: tanh(dot(input,W) + b)
		
		ARGS:
			rng(np.random.RandomState): a random number generator used
				to initialize weights
			input(T.dmatrix): a symbolic tensor of shape (n_examples, n_in)
			n_in (int): dimensionality of input
			n_out (int): number of hidden units (ie. output from this layer)
			activation(theano.Op or function): Non linearity to be applied
				in the hidden layer
		
		"""
		self.input = input
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
		
		lin_output = T.dot(input, self.W) + self.b
		self.output = (
			lin_output if activation is None
			else activation(lin_output)
		)
		# parameters of the model
		self.params = [self.W, self.b]
