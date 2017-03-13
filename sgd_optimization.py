import sys
import timeit
import six.moves.cPickle as pickle

import numpy as np

import theano
import theano.tensor as T
import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu0')

import LogicRegression as lr

class SgdOptimization:
	"""SGD optimization of a log-linear model
		
	This is demonstrated on MNIST
	"""	
	
	def __init__(self,
			datasets,
			learning_rate=0.13,
			n_epochs=1000,
			batch_size=1000):
		
		"""Initialise optimisation
		
		Args:
			datasets (dict): training, validation and test datasets
				each is dict of:
				input data, presented as m rasterized images
				targets, presented as m 1D vectors of [int] labels
			learning_rate (float): learning rate used
				(factor for the stochastic gradient)
			n_epochs (int): max number of epochs to run the optimizer

		"""
		
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batch_size = batch_size

		# store inputs and targets for training, validation and test
		# calls shared_datasets to store them as shared variables
		self.X_train, self.y_train = self.shared_dataset(datasets[0])
		self.X_valid, self.y_valid = self.shared_dataset(datasets[1])
		self.X_test, self.y_test = self.shared_dataset(datasets[2])
			
		print('train inputs: {}'.format(self.X_train.get_value(
			borrow=True).shape))
		print('train labels: {}'.format(self.y_train.shape))
		print('validation inputs: {}'.format(self.X_valid.get_value(
			borrow=True).shape))
		print('validation labels: {}'.format(self.y_valid.shape))
		print('test inputs: {}'.format(self.X_test.get_value(
			borrow=True).shape))
		print('test labels: {}'.format(self.y_test.shape))
		
		print(type(self.X_train))
	
		# number of minibatches for training, validation and test
		self.n_train_batches = self.X_train.get_value(
			borrow=True).shape[0] // batch_size
		self.n_valid_batches = self.X_valid.get_value(
			borrow=True).shape[0] // batch_size
		self.n_test_batches = self.X_test.get_value(
			borrow=True).shape[0] // batch_size
	
		print('number of minibatches of size {}: {}, {}, {}'.format(
			self.batch_size, self.n_train_batches,
			self.n_valid_batches, self.n_test_batches))
	
		# number of training cases
		self.m = self.X_train.get_value(
			borrow=True).shape[0]
		print('number of training cases: {}'.format(self.m))
	
	def shared_dataset(self, data_xy, borrow=True):
		"""Load the dataset into shared variables
		
		This allows Theano to copy it into the GPU memory
		(when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x,
			dtype=theano.config.floatX),
			borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y,
			dtype=theano.config.floatX),
			borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ‘‘floatX‘‘ as well
		# (‘‘shared_y‘‘ does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn’t make sense) therefore instead of returning
		# ‘‘shared_y‘‘ we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')	
	
	def build(self):
		"""Build actual model
		
		"""
		print('... building the model')
		
		# allocate symbolic variables for the data
		index = T.lscalar()  # index to a [mini]batch
		
		mb_start = index * self.batch_size
		mb_end = (index + 1) * self.batch_size
		
		# generate symbolic variables: inputs x_mb,i, targets y_mb,o
		x = T.matrix('x')
		y = T.ivector('y')
		
		# construct the logistic regression class
		# Each MNIST image has size 28*28
		# x is outside LogisticRegression class and needs to be passed
		self.classifier = lr.LogisticRegression(input=x, n_in=28 * 28, n_out=10)
		
		# cost to be minimised
		# negative log likelihood of the model in symbolic format
		# x is an implicit symbolic input because it was defined at init
		cost = self.classifier.negative_log_likelihood(y)
		
		# compiling a Theano function that computes the mistakes
		# that are made by the model on a minibatch
		self.test_model = theano.function(
		    inputs=[index],
		    outputs=self.classifier.errors(y),
		    givens={
		        x: self.X_test[mb_start: mb_end],
		        y: self.y_test[mb_start: mb_end]
		    }
		)
		self.validate_model = theano.function(
		    inputs=[index],
		    outputs=self.classifier.errors(y),
		    givens={
		        x: self.X_valid[mb_start: mb_end],
		        y: self.y_valid[mb_start: mb_end]
		    }
		)
			
		# automatic differentiation to compute the gradient of
		# cost with respect to theta = (W,b), dl/dW and dl/db
		g_W = T.grad(cost=cost, wrt=self.classifier.W)
		g_b = T.grad(cost=cost, wrt=self.classifier.b)
		
		# start-snippet-3
		# specify how to update the parameters of the model
		# updates is list of paris:
		# first element is symbolic variable to be updated
		# second element is symbolic function for calculating its new value
		updates = [(self.classifier.W, self.classifier.W - self.learning_rate * g_W),
		           (self.classifier.b, self.classifier.b - self.learning_rate * g_b)]
		
		# compiling a Theano function ‘train_model‘
		# input is mini-batch index --> X and y
		# returns the cost
		# on every function call it:
		# - replaces X and y with the slices
		#   from the training set specified by index
		# - evaluates the cost
		# - applies the operation defined by the updates list
		# Thus each time train_model(index) is called, it will compute
		# and return the cost of a minibatch, while also performing
		# a step of MSGD.
		# The  entire learning algorithm thus consists in looping over
		# all examples in the dataset, considering all the examples in
		# one minibatch at a time, and repeatedly calling
		# the train_model function
		self.train_model = theano.function(
		    inputs=[index],
		    outputs=cost,
		    updates=updates,
		    givens={
		        x: self.X_train[mb_start: mb_end],
		        y: self.y_train[mb_start: mb_end]
		    }
		)
		# end-snippet-3
	
	def train(self):
		"""Train model
		
		"""		
		print('... training the model')
		
		# early-stopping parameters
		# look at this many examples regardless
		patience = 5000
		# wait this much longer when a new best is found
		patience_increase = 2
		# a relative improvement of this much is considered significant
		improvement_threshold = 0.995
		# go through this many minibatches before checking
		# the network on the validation set;
		# should be less than the patience
		# in this case we check every epoch
		validation_frequency = min(self.n_train_batches, patience / 2)
		best_validation_loss = np.inf
		test_score = 0.
		start_time = timeit.default_timer()
		done_looping = False
		epoch = 0
		
		while (epoch < self.n_epochs) and (not done_looping):
			epoch += 1
			for minibatch_index in range(self.n_train_batches):
				minibatch_avg_cost = self.train_model(minibatch_index)
				# iteration number
				iter = (epoch - 1) * self.n_train_batches + minibatch_index
				if (iter + 1) % validation_frequency == 0:
					# compute zero-one loss on validation set
					validation_losses = [self.validate_model(i)
						for i in range(self.n_valid_batches)]
					this_validation_loss = np.mean(validation_losses)
					print(
						(
							'epoch {}, minibatch {}/{},'
							' training loss {}'
						).format(
							epoch, minibatch_index + 1, self.n_train_batches,
							minibatch_avg_cost
						)
					)
					print(
						(
							'epoch {}, minibatch {}/{},'
							' validation error {:.5f}%'
						).format(
							epoch, minibatch_index + 1, self.n_train_batches,
							this_validation_loss * 100.
						)
					)
					# if we got the best validation score until now
					if this_validation_loss < best_validation_loss:
						# improve patience if loss improvement is good enough
						if this_validation_loss < best_validation_loss * \
							improvement_threshold:
							patience = max(patience, iter * patience_increase)
						best_validation_loss = this_validation_loss 
						# test it on the test set
						test_losses = [self.test_model(i)
							for i in range(self.n_test_batches)]
						test_score = np.mean(test_losses)
						print(
							(
								'epoch {}, minibatch {}/{},'
								' test error of best model {:.5f}%'
							).format(
								epoch, minibatch_index + 1,
								self.n_train_batches, test_score * 100.
							)
						)
						# save the best model
						with open('best_model.pkl', 'wb') as f:
							pickle.dump(self.classifier, f)
		
					if patience <= iter:
						done_looping = True
						break
		
		end_time = timeit.default_timer()
		print(
			(
				'Optimization complete with best validation score'
				' of {:.5f}, with test performance {:.5f}'
			).format(
				best_validation_loss * 100., test_score * 100.
			)
		)
		
		print('The code run for {} epochs, with {:.5f} epochs/sec'.format(
			epoch, 1. * epoch / (end_time - start_time)))
		
		
