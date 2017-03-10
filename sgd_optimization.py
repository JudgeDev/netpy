import sys
import timeit
import numpy as np

import theano
import theano.tensor as T

import LogicRegression

class SgdOptimization:
	"""SGD optimization of a log-linear model
		
	This is demonstrated on MNIST
	"""	
	
	def __init__(self,
			datasets,
			learning_rate=0.13,
			n_epochs=1000,
			batch_size=600):
		
		"""Initialise optimisation
		
		Args:
			learning_rate (float): learning rate used
				(factor for the stochastic gradient)
			n_epochs (int): max number of epochs to run the optimizer
			datasets (dict): training, validation and test datasets
				each is dict of:
				input data, presented as m rasterized images
				targets, presented as m 1D vectors of [int] labels
		 
		"""
		
		self.lerning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		
		# store inputs and targets for training, validation and test
		self.X_train, self.y_train = datasets[0]
		self.X_valid, self.y_valid = datasets[1]
		self.X_test, self.y_test = datasets[2]
			
		print('train inputs: {}'.format(self.X_train.shape))
		print('train labels: {}'.format(self.y_train.shape))
		print('validation inputs: {}'.format(self.X_valid.shape))
		print('validation labels: {}'.format(self.y_valid.shape))
		print('test inputs: {}'.format(self.X_test.shape))
		print('test labels: {}'.format(self.y_test.shape))
	
		# number of minibatches for training, validation and test
		self.n_train_batches = self.X_train.shape[0] // batch_size
		self.n_valid_batches = self.X_valid.shape[0] // batch_size
		self.n_test_batches = self.X_test.shape[0] // batch_size
	
		print('number of minibatches of size {}: {}, {}, {}'.format(
			self.batch_size, self.n_train_batches,
			self.n_valid_batches, self.n_test_batches))
	
		# number of training cases
		self.m = self.X_train.shape[0]
		print('number of training cases: {}'.format(self.m))
	
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
		classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
		
		# cost to be minimised
		# negative log likelihood of the model in symbolic format
		# x is an implicit symbolic input because it was defined at init
		cost = classifier.negative_log_likelihood(y)
		
		# compiling a Theano function that computes the mistakes
		# that are made by the model on a minibatch
		test_model = theano.function(
		    inputs=[index],
		    outputs=classifier.errors(y),
		    givens={
		        x: X_test[mb_start: mb_end],
		        y: y_test[mb_start: mb_end]
		    }
		)
		validate_model = theano.function(
		    inputs=[index],
		    outputs=classifier.errors(y),
		    givens={
		        x: X_valid[mb_start: mb_end],
		        y: y_valid[mb_start: mb_end]
		    }
		)
			
		# automatic differentiation to compute the gradient of
		# cost with respect to theta = (W,b), dl/dW and dl/db
		g_W = T.grad(cost=cost, wrt=classifier.W)
		g_b = T.grad(cost=cost, wrt=classifier.b)
		
		# start-snippet-3
		# specify how to update the parameters of the model
		# updates is list of paris:
		# first element is symbolic variable to be updated
		# second element is symbolic function for calculating its new value
		updates = [(classifier.W, classifier.W - self.learning_rate * g_W),
		           (classifier.b, classifier.b - self.learning_rate * g_b)]
		
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
		train_model = theano.function(
		    inputs=[index],
		    outputs=cost,
		    updates=updates,
		    givens={
		        x: train_set_x[mb_start: mb_end],
		        y: train_set_y[mb_start: mb_end]
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
				minibatch_avg_cost = train_model(minibatch_index)
				# iteration number
				iter = (epoch - 1) * self.n_train_batches + minibatch_index
				if (iter + 1) % validation_frequency == 0:
					# compute zero-one loss on validation set
					validation_losses = [validate_model(i)
						for i in range(self.n_valid_batches)]
					this_validation_loss = np.mean(validation_losses)
					print(
						(
							'epoch {}, minibatch {}/{},'
							' validation error {:.5f}%'
						).format(
							epoch, minibatch_index + 1, n_train_batches,
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
						test_losses = [test_model(i)
							for i in range(n_test_batches)]
						test_score = np.mean(test_losses)
						print(
							(
								'epoch {}, minibatch {}/{},'
								' test error of best model {:.5f}%'
							).format(
								epoch, minibatch_index + 1,
								n_train_batches, test_score * 100.
							)
						)
						# save the best model
						with open('best_model.pkl', 'w') as f:
							cPickle.dump(classifier, f)
		
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
		
		print('The code run for {} epochs, with {} epochs/sec'.format(
			epoch, 1. * epoch / (end_time - start_time)))
		
		#print >> sys.stderr, (’The code for file ’ + os.path.split(__file__)[1] + ’ ran for %.1fs’ % ((end_time - start_time)))
		
