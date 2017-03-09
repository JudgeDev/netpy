import sys
import timeit
import numpy as np

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
		
		:type learning_rate: float
		:param learning_rate: learning rate used (factor for the stochastic gradient)
		:type n_epochs: int
		:param n_epochs: maximal number of epochs to run the optimizer
		:type datasets: string
		:param datasets: training, validation and test datasets 
		 
		"""
		
		self.lerning_rate = learning_rate
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		
		# store inputs and targets for training, validation and test
		self.X, self.y = datasets[0]
		self.Xv, self.yv = datasets[1]
		self.Xt, self.yt = datasets[2]
			
		print('train inputs: {}'.format(self.X.shape))
		print('train labels: {}'.format(self.y.shape))
		print('validation inputs: {}'.format(self.Xv.shape))
		print('validation labels: {}'.format(self.yv.shape))
		print('test inputs: {}'.format(self.Xt.shape))
		print('test labels: {}'.format(self.yt.shape))
	
		# number of minibatches for training, validation and test
		self.n_train_batches = self.X.shape[0] // batch_size
		self.n_valid_batches = self.valid_set_x.shape[0] // batch_size
		self.n_test_batches = self.test_set_x.shape[0] // batch_size
	
		print('number of minibatches of size {}: {}, {}, {}'.format(
		self.batch_size, self.n_train_batches, self.n_valid_batches, self.n_test_batches))
	
		# number of training cases
		self.m = self.X.shape[0]
		print('number of training cases: {}'.format(self.m))
		
		
		######################
		# BUILD ACTUAL MODEL
		# ######################
		print('... building the model')
		print('Need to add model build code')
		"""
		@TODO do something with Theano code
		
		# allocate symbolic variables for the data
		index = T.lscalar() # index to a [mini]batch
		
		# generate symbolic variables for input (x and y represent a
		# minibatch)
		x = T.matrix(’x’) # data, presented as rasterized images
		y = T.ivector(’y’) # labels, presented as 1D vector of [int] labels
		
		# construct the logistic regression class
		# Each MNIST image has size 28*28
		classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
		
		# the cost we minimize during training is the negative log likelihood of the model in symbolic format
		cost = classifier.negative_log_likelihood(y)
		
		# compiling a Theano function that computes the mistakes that are made by the model on a minibatch
		test_model = theano.function(
		    inputs=[index],
		    outputs=classifier.errors(y),
		    givens={
		        x: test_set_x[index * batch_size: (index + 1) * batch_size],
		        y: test_set_y[index * batch_size: (index + 1) * batch_size]
		    }
		)
		validate_model = theano.function(
		    inputs=[index],
		    outputs=classifier.errors(y),
		    givens={
		        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
		        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		    }
		)
			
		# compute the gradient of cost with respect to theta = (W,b)
		g_W = T.grad(cost=cost, wrt=classifier.W)
		g_b = T.grad(cost=cost, wrt=classifier.b)
		
		# start-snippet-3
		# specify how to update the parameters of the model as a list of # (variable, update expression) pairs.
		updates = [(classifier.W, classifier.W - learning_rate * g_W),
		           (classifier.b, classifier.b - learning_rate * g_b)]
		
		# compiling a Theano function ‘train_model‘ that returns the cost, but in the same time updates the parameter of the model based on the rules defined in ‘updates‘
		train_model = theano.function(
		    inputs=[index],
		    outputs=cost,
		    updates=updates,
		    givens={
		        x: train_set_x[index * batch_size: (index + 1) * batch_size],
		        y: train_set_y[index * batch_size: (index + 1) * batch_size]
		    }
		)
		# end-snippet-3
		"""
	
	
	def train(self):
				
		###############
		# TRAIN MODEL #
		###############
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
		# in this case we check every epoch
		validation_frequency = min(self.n_train_batches, patience / 2)
		best_validation_loss = np.inf
		test_score = 0.
		start_time = timeit.default_timer()
		done_looping = False
		epoch = 0
		
		while (epoch < self.n_epochs) and (not done_looping):
			epoch = epoch + 1
			"""
			for minibatch_index in xrange(self.n_train_batches):
				# minibatch_avg_cost = train_model(minibatch_index)
				# iteration number
				iter = (epoch - 1) * n_train_batches + minibatch_index
				if (iter + 1) % validation_frequency == 0:
					# compute zero-one loss on validation set
					validation_losses = [validate_model(i)
						for i in xrange(n_valid_batches)]
					this_validation_loss = numpy.mean(validation_losses)
					print(
						'epoch {}, minibatch {}/{}, validation error {}',format(
							epoch,minibatch_index +1, n_train_batches, this_validation_loss * 100.)
					)
					# if we got the best validation score until now
					if this_validation_loss < best_validation_loss:
						# improve patience if loss improvement is good enough
						if this_validation_loss < best_validation_loss * \
							improvement_threshold:
							patience = max(patience, iter * patience_increase)
						best_validation_loss = this_validation_loss # test it on the test set
						test_losses = [test_model(i)
							for i in xrange(n_test_batches)]
						test_score = numpy.mean(test_losses)
						print(
							(
								'epoch %i, minibatch %i/%i, test error of'
								'best model %f %%'
							).format(epoch, minibatch_index + 1, n_train_batches, test_score * 100.)
						)
						# save the best model
						with open(’best_model.pkl’, ’w’) as f:
							cPickle.dump(classifier, f)
		
					if patience <= iter:
						done_looping = True
						break
		"""
		
		end_time = timeit.default_timer()
		print(
			(
			'Optimization complete with best validation score of {},'
			' with test performance {}'
			).format(best_validation_loss * 100., test_score * 100.)
		)
		
		print('The code run for {} epochs, with {} epochs/sec'.format(
			epoch, 1. * epoch / (end_time - start_time)))
		
		#print >> sys.stderr, (’The code for file ’ + os.path.split(__file__)[1] + ’ ran for %.1fs’ % ((end_time - start_time)))
		
