"""Logistic regression class

Logistic regression is a probabilistic, linear classifier.
It is parametrized by:
a weight matrix :math:‘W‘ and
a bias vector :math:‘b‘.

Classification is done by projecting data points onto
a set of hyperplanes, the distance to which is used to determine
a class membership probability.

Mathematically, this can be written as:
.. math::
P(Y=i|x, W,b) &= softmax_i(W x + b) \\
				&= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}
The output of the model or prediction is then done by taking
the argmax of the vector whose i’th element is P(Y=i|x).
.. math::
y_{pred} = argmax_i P(Y=i|x,W,b)

References:
- textbooks: "Pattern Recognition and Machine Learning" -
Christopher M. Bishop, section 4.3.2
"""
__docformat__ = 'restructedtext en'

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu0')

class LogisticRegression (object):
	"""Multi-class Logistic Regression Class

	The logistic regression is fully described by a weight matrix :math:‘W‘
	and bias vector :math:‘b‘. Classification is done by projecting data
	points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""

	def __init__(self, input, n_in, n_out):
		""" Initialize the parameters of the logistic regression
		
		Args:
			input (T.TensorType): symbolic variable that describes the
				input of the architecture (one minibatch)
			n_in(int): number of input units, the dimension of the space in
				which the datapoints lie
			n_out(int): number of output units, the dimension of the space
				in which the labels lie
		"""
		# start-snippet-1
		# Since the parameters of the model must maintain a persistent state
		# throughout training, we allocate shared variables for W,b.
		# This declares them both as being symbolic Theano variables,
		# but also initializes their contents.
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(
			value=np.zeros(
				(n_in, n_out),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)
		# initialize the biases b as a vector of n_out 0s
		self.b = theano.shared(
			value=np.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)
		
		# symbolic expression for computing the matrix of class-membership
		# probabilities. where:
		# W is a matrix where column-k represent the separation hyperplane
		#  for class-k
		# x is a matrix where row-j  represents input training sample-j
		# b is a vector where element-k represent the free parameter of
		# 	hyperplane-k
		# The dot and softmax operators are then used to compute the vector
		# P(Y|x,Wb)).
		# The result p_y_given_x is a symbolic variable of vector-type
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		
		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		# Uses T.argmax operator, which will return the index at which
		# p_y_given_x is maximal (i.e. the class with maximum probability)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1
		
		# parameters of the model
		self.params = [self.W, self.b]
		
		# keep track of model input
		self.input = input
	
	def negative_log_likelihood(self, y):
		"""Return the negative log-likelihood of the prediction
		
		Args:
			y (T.TensorType): a vector that gives for each example the
		correct label
		Returns:
			float: mean of the negative log-likelihood of the prediction
				of this model under a given target distribution
		
		.. math::
		
		\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
		\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
		\log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
		\ell (\theta=\{W,b\}, \mathcal{D})
				
		Note: we use the mean instead of the sum so that the learning
		rate is less dependent on the batch size
		"""
		# start-snippet-2
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1]
		# T.log(self.p_y_given_x) is a matrix of Log-Probabilities
		# with one row per example and one column per class
		# LP_n,L 
		# LP_n,L[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]]
		# T.mean(LP[T.arange(y.shape[0]),y]) is the mean
		# (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		# end-snippet-2
	
	def errors(self, y):
		"""Return zero-one loss error
		
		Args:
			y(T.TensorType): vector that gives for each example the
				correct label
		Returns:
			float: number of errors in the minibatch over the total
				number of examples of the minibatch
				
		l_0,1 = sum_{i=0,D}I_{f(x)^i!=y^i}
		
		"""
		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()
