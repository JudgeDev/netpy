import math
from collections import defaultdict

import numpy as np
from net import NetPy


class TorontoNet:
	"""Python version of assignment 3 in Neural Networks course
	
	"""
	def __init__(self,
		# net parameters
		n_in, n_hid, n_out):
		"""Set up net
		
		Includes:
			setting weight parameters
		"""	 
		print('Initializing TorontoNet')
		self.n_in = n_in
		self.n_out = n_out
		print('number of input layers: {}'.format(n_in))
		print('number of hidden layers: {}'.format(n_hid))
		print('number of output layers: {}'.format(n_out))
		# set up initial model
		self.model = self.initial_model(n_hid)
			
	def optimization(self,
		datas,
		n_iters, mini_batch_size, learning_rate, momentum_multiplier,
		wd_coefficient, do_early_stopping):
		"""Train and optimize net
		
		Args:
			datasets (dict): train, validation and test data
			n_iters (int): number of iterations
			mini_batch_size (int): number of cases in mini batch
			learning_rate (float):
			momentum_multiplier (float):
			wd_coefficient (float): weight decay
			do_early_stopping (bool): early stopping flag
		
		The original code used data sets with dimension inputs*cases
		(i.e. _i,m)
			
		"""
		print('Optimizing TorontoNet')
		
		print('train inputs: {}'.format(datas['training']['inputs'].shape))
		print('train labels: {}'.format(datas['training']['targets'].shape))
		print('valdidation inputs: {}'.format(datas['validation']['inputs'].shape))
		print('valdidation labels: {}'.format(datas['validation']['targets'].shape))
		print('test inputs: {}'.format(datas['test']['inputs'].shape))
		print('test labels: {}'.format(datas['test']['targets'].shape))
		
		n_training_cases = datas['training']['inputs'].shape[1]
		print('number of training cases: {}'.format(n_training_cases))
		
		if n_iters != 0:
			self.test_gradient(self.model, datas['training'], wd_coefficient)

		theta = self.model_to_theta(self.model)
		momentum_speed = theta * 0
		training_data_losses = []
		validation_data_losses = []
		training_batch = {}  # dictionary for mini batch data
		best_so_far = {}  # dict of best values
		if do_early_stopping:
			best_so_far['theta'] = -1  # this will be overwritten soon
			best_so_far['validation_loss'] = np.inf
			best_so_far['after_n_iters'] = -1
		for optimization_iteration_i in range(n_iters):
			
			model = self.theta_to_model(theta)

			training_batch_start = (
				(optimization_iteration_i * mini_batch_size) %
				n_training_cases)
			training_batch['inputs'] = datas['training']['inputs'][:,
				training_batch_start : training_batch_start + mini_batch_size]
			training_batch['targets'] = datas['training']['targets'][:,
				training_batch_start : training_batch_start + mini_batch_size]
			
			gradient = self.model_to_theta(
				self.d_loss_by_d_model(
					model, training_batch, wd_coefficient
					)
			)
			momentum_speed = momentum_speed * momentum_multiplier - gradient
			theta = theta + momentum_speed * learning_rate

			model = self.theta_to_model(theta)
				
			if ((optimization_iteration_i + 1) % np.round(n_iters/10)) == 0:
				# temp move from above if statement
				# only tests losses every 1/10 number of iters
				training_data_losses.append(self.loss(model, datas['training'], wd_coefficient))
				validation_data_losses.append(self.loss(model, datas['validation'], wd_coefficient))
				if do_early_stopping and validation_data_losses[-1] < best_so_far['validation_loss']:
					best_so_far['theta'] = theta # this will be overwritten soon
					best_so_far['validation_loss'] = validation_data_losses[-1]
					best_so_far['after_n_iters'] = optimization_iteration_i + 1

				print(
				(
					'After {} optimization iterations:\n'
					'Training data loss is {:.5f}, and'
					' validation data loss is {:.5f}\n'
				).format(
					optimization_iteration_i + 1,
					training_data_losses[-1],
					validation_data_losses[-1])
				)
				
		#if n_iters ~= 0, test_gradient(model, datas.training, wd_coefficient); end # check again, this time with more typical parameters
		if do_early_stopping:
			print(
				(
					'Early stopping: validation loss was lowest'
					'  after {} iterations. We chose the model that'
					' we had then.\n'
				).format(best_so_far['after_n_iters'])
			)
			theta = best_so_far['theta']

		# the optimization is finished. Now do some reporting.
		model = self.theta_to_model(theta)
		if n_iters != 0:
			# Plot
			NetPy().draw_graph('Losses',
				['$iteration number$', '$loss$'], 
				[(range(len(training_data_losses)), training_data_losses, 'b', 'training'),
				(range(len(validation_data_losses)), validation_data_losses, 'r', 'validation')],
				grid=True, legend=True
			)

		for data_name, data_segment in datas.items():
			loss_wd = self.loss(model, data_segment, wd_coefficient)
			print('\nThe loss on the {} data is {:.5f}'.format(
				data_name, loss_wd))
			if wd_coefficient != 0:
				loss = self.loss(model, data_segment, 0)
				print(
					(
						'The classification loss (i.e. without weight'
						' decay) on the {} data is {:.5f}'
					).format(data_name, loss))
			err = self.classification_performance(model, data_segment)
			print(
				(
					'The classification error rate on the {} data'
					' is {:.5f} ({:.2f}%)\n'
				).format(data_name, err, (1 - err) * 100))

	def test_gradient(self, model, data, wd_coefficient):
		"""Check some gradient values
		
		Test the gradient not for every element of theta,
		because that's a lot of work. Test for only a few elements.
		"""
		base_theta = self.model_to_theta(model)
		h = 1e-2
		correctness_threshold = 1e-5
		analytic_gradient = self.model_to_theta(self.d_loss_by_d_model(model, data, wd_coefficient));
		for i in range(10):
			# 1299721 is prime and thus ensures a somewhat random-like
			# selection of indices
			test_index = ((i+1) * 1299721 % base_theta.size)
			print('Gradient check: test index {}'.format(test_index))
			analytic_here = analytic_gradient[test_index]
			theta_step = base_theta * 0
			theta_step[test_index] = h
			contribution_distances = [-4, -3, -2, -1, 1, 2, 3, 4]
			contribution_weights = [
				1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280]
			temp = 0;
			for distance, weight in zip(contribution_distances,
				contribution_weights):
				temp += self.loss(self.theta_to_model(
					base_theta + theta_step * distance),
					data, wd_coefficient) * weight
			fd_here = temp / h
			diff = abs(analytic_here - fd_here)
			# fprintf('%d %e %e %e %e\n', test_index, base_theta(test_index), diff, fd_here, analytic_here);
			if diff < correctness_threshold: continue
			if diff / (abs(analytic_here) + abs(fd_here)) < correctness_threshold: continue
			raise Exception(
				(
					'Theta element {}, with value {},'
					' has finite difference gradient {}'
					' but analytic gradient {}.'
					' That looks like an error.\n'
				).format(test_index, base_theta[test_index],
					fd_here, analytic_here)
			)
		print(
			(
				'Gradient test passed. That means that the gradient'
				' that your code computed is within 0.001%% of'
				' the gradient that the finite difference approximation'
				' computed, so the gradient calculation procedure'
				' is probably correct (not certainly, but probably).\n'
			)
		)

	def log_sum_exp_over_rows(self, a):
		"""Compute log(sum(exp(a), 1)) in a numerically stable way
		
		Args:
			a (matrix): _parameters,cases
		
		Returns:
			matrix: log(sum_rows(exp(a)))_1,cases
		"""
		# get the vax value in each column --> row of max column values
		max_cols = a.max(0) 
		# remove max col values before taking exp (using broadcasting)
		# and add back after log and sum (of cols) gives log(sum(exp(a)))
		return np.log(np.exp(a - max_cols).sum(axis=0)) + max_cols


	def loss(self, model, data, wd_coefficient):
		"""Return loss for given model, data and weight decay
		Args:
			model (dict): dictionary of layer weight matrices
			model['input_to_hid'] is a matrix of size
				<number of hidden units> by <number of inputs i.e. 256>
				It contains the weights from the input units
				to the hidden units
			model['hid_to_class'] is a matrix of size
				<number of classes i.e. 10> by <number of hidden units>
				It contains the weights from the hidden units
				to the softmax units
			data (dict): training inputs
			data['inputs'] is a matrix of size
				<number of inputs i.e. 256> by <number of data cases>
				Each column describes a different data case
			data['targets'] is a matrix of size
				<number of classes i.e. 10> by <number of data cases>
				Each column describes a different data case
				It contains a one-of-N encoding of the class,
				i.e. one element in every column is 1 and
				the others are 0
			wd_coefficient (float): weight decay coefficient
		
		Returns:
			float: total loss
		"""
		# Before we can calculate the loss, we need to calculate
		# a variety of intermediate values, like the state of
		# the hidden units
		# input to the hidden units, i.e. before the logistic:
		# zh = wh * i
		# size: <number of hidden units> by <number of data cases>
		hid_input = model['input_to_hid'].dot(data['inputs'])
  
		# output of the hidden units, i.e. after the logistic:
		# yh = sig(zh)
		# size: <number of hidden units> by <number of data cases>
		hid_output = NetPy().logistic(hid_input)
  
		# input to the components of the softmax: zo = wo * yh
		# size: <number of classes, i.e. 10> by <number of data cases>
		class_input = model['hid_to_class'].dot(hid_output)

		# The following three lines of code implement the softmax.
		# Normally, a softmax is described as exponential divided by
		# a sum of exponentials - y_i = exp(z_i) / sum_j(exp(z_j)).
		# What we do here is exactly equivalent:
		# exp(z_i - log(sum_j(exp(z_j))))
		# The second term is calculated in log_sum_exp_over_rows()
		# This is more numerically stable because there will never be
		# really big numbers involved.
		# The exponential in the lectures can lead to really big numbers,
		# which are fine in mathematical equations,
		# but can lead problems in prgrams.
		# Programs may not deal well with really large numbers,
		# like the number 10 to the power 1000.
		# Computations with such numbers get unstable, so we avoid them.

		# log(sum(exp of class_input)) is what we subtract to get
		# properly normalized log class probabilities.
		# size: <1> by <number of data cases>
		class_normalizer = self.log_sum_exp_over_rows(class_input)
		 
		# log of probability of each class.
		# size: <number of classes, i.e. 10> by <number of data cases>
		# use broadcasting since number of cols (cases) is same
		log_class_prob = class_input - class_normalizer 
		# probability of each class. Each column (i.e. each case) sums to 1.
		# size: <number of classes, i.e. 10> by <number of data cases>
		class_prob = np.exp(log_class_prob)

		# select the right log class probability using that sum;
		# cross-entropy cost function:
		# CE = -sum_j(t_j * log(y_j))
		# then take the mean over all data cases.
		classification_loss = - np.mean(
			(log_class_prob * data['targets']).sum(axis=0))

		# weight decay loss. very straightforward:
		# E = 1/2 * wd_coefficient * theta^2
		wd_loss = (self.model_to_theta(model)**2).sum() / 2 * wd_coefficient

		return classification_loss + wd_loss
	
	def d_loss_by_d_model(self, model, data, wd_coefficient):
		"""Calculate the gradients from test data and the weight decay
		
		Args:
			model (dict): dictionary of layer weight matrices
			model['input_to_hid'] is a matrix - w^i_h,i
				<number of hidden units> by <number of inputs i.e. 256>
				It contains the weights from the input units
				to the hidden units
			model['hid_to_class'] is a matrix - w^h_c,h
				<number of classes i.e. 10> by <number of hidden units>
				It contains the weights from the hidden units
				to the softmax units
			data (dict): training inputs
			data['inputs'] is a matrix - i_i,d
				<number of inputs i.e. 256> by <number of data cases>
				Each column describes a different data case
			data['targets'] is a matrix - t_c,d
				<number of classes i.e. 10> by <number of data cases>
				Each column describes a different data case
				It contains a one-of-N encoding of the class,
				i.e. one element in every column is 1 and
				the others are 0
			wd_coefficient (float): weight decay coefficient
		
		Returns:
			dict: gradients in same form as parameter <model>
				i.e. it has matrices input_to_hid and ret.hid_to_class

		"""
		#calculate states:
		# @TODO - should be a function as it is repeated in
		# function loss()

		# input to the hidden units, i.e. before the logistic:
		# z^h_h,d = w^i_h,i * i_i,d
		hid_input = model['input_to_hid'].dot(data['inputs'])

		# output of the hidden units, i.e. after the logistic:
		# y^h_h,d = sig(z^h_h,d)
		hid_output = NetPy().logistic(hid_input)

		# input to the components of the softmax:
		# z^o_c,d = w^h_c,h * y^h_h,d
		class_input = model['hid_to_class'].dot(hid_output)

		# output of the softmax:
		# y^o_c,d = exp(z^o_c,d) / sum_C(exp(z^o_c,d))

		# this is quivalent to:
		# exp(z^o - log(sum_C(exp(z^o))))
		# The second term is calculated in log_sum_exp_over_rows()
		# This is more numerically stable because there will never be
		# really big numbers involved.

		# normalizer to subtract from each log class probability:
		# sum_C(exp(z^o))_1,D
		class_normalizer = self.log_sum_exp_over_rows(class_input)
		# normalized classes = log of probability of each class:
		# log(y^0)_c,d
		log_class_prob = class_input - class_normalizer 
		# probability of each class:
		# log(y^0)_c,d
		# Each column (i.e. each case) sums to 1.
		class_prob = np.exp(log_class_prob)

		numcases = data['inputs'].shape[1]  # number of data cases, D

		# (1) for general layer (L3 - Learning the weights)
		# (1a) z^l_j = b^{l-1}_i + sum_j(w^{l-1}_j,i.y^{l-1}_i); input or logit
		# (1b) dz^l/dw^{l-1} = y^{l-1}
		# (1c) dz^l/dy^{l-1} = w^{l-1}

		# (2) for neuron (L3 - Learning the weights)
		# (2a) y = sig(z) = 1 / (1 + exp(-z)); output or activation
		# (2b) dy/dz = y(1-y)

		# (3) for softmax (L4)
		# (3a) y = exp(z) / sum_C(exp(z)); C is number of classes
		# (3b) dy/dz = y(1-y)
		# (3c) CE = -sum_C(t.log(y)); CE is cost or cross entropy
		# (3d) dCE/dz = y - t

		# (4) L2 weight penalty (L9 - Limiting the size of the weights)
		# (4a) L  = E + wd/2.sum_C((w)^2); wd is weight decay or weight cost
		# (4b) dL/dw = dE/dw + wd.w; weight decay penalises large inactive weights

		# Hinton's backpropagation rules
		# gradient of input of a unit in terms of its output
		# (h1) dE/dz^l = dy^l/dz^l.dE/dy^l
		# gradient of output of unit in terms of input to unit in next layer
		# (h2) dE/dy^l-1 = sum_j(dz^l/dy^l-1.dE/dz^l) = sum_j(w^l-1_j_i.dE/dz^l_j)
		# gradient of weight in terms of input to next layer
		# (h3) dE/dw^l = dz^l/dw^l-1.dE/dz^l = y^l-1.dE/dz^l

		# application to present model
		# overall model
		# M = {w^i_h,i, w^h_c,h}
		# gradient of cost for overall model
		# dL/dM = {dL/dw^i_h,i, dL/dw^h_c,h}

		# gradient of hidden to class weights with weight penalty wd
		# dL/dw^h_c,h = dE/dw^h_c,h + wd.w^h_c,h; = (4b)
		# gradient of cost of output layer - o
		# dE/dw^h_c,h = dE/dz^o_c.dz^o_c/dw^h_c,h; chain rule, (h3)
		# for cross-entropy cost function
		# = (y^o_c - t^o_c).y^h_h; (3d) and (1b) or (h3)
		# Thus dL/dw^h_c,h = (y^o_c - t^o_c).y^h_h + wd.w^h_c,h
		# normalising the term containing the data cases
		hid_to_class = (
			((class_prob - data['targets']).dot(hid_output.T))
			/ numcases + wd_coefficient * model['hid_to_class']
		)

		# gradient of the input to hidden weights with weight penalty wd
		# dL/dw^i_h,i = dE/dw^i_h,i + wd.w^i_h,i; = (4b)
		# gradient of cost function of hidden layer - h
		# dE/dw^i_h,i = dE/dz^h_h.dz^h_h/dw^i_h,i; chain rule, (h3)
		# = dE/dz^h_h.i_i; (1b) or (h3)
		# gradient in terms of output of hidden layer
		# dE/dz^h_h = dE/dy^h_h.dy^h_h/dz^h; (h1)
		# for sigmoid
		# = dE/dy^h_h.y^h_h.(1-y^h_h); (2b) or (h1)
		# gradient in terms of input to unit in next layer
		# dE/dy^h_h = sum_c(dE/dz^o_c.dz^o_c/dy^h_h) = dE/dz^o_c*w^h_c,h; (h2)
		# Thus dL/dw^i_h,i = (y^o_c - t^o_c)*w^h_c,h.y^h_h(1-y^h_h).i_i + wd.w^i_h,i
		# normalising the term containing the data cases
		input_to_hid = (
			((model['hid_to_class'].T.dot(class_prob - data['targets'])
			* hid_output * (1 - hid_output)).dot(data['inputs'].T))
			/ numcases + wd_coefficient * model['input_to_hid']
		)
		return {'input_to_hid': input_to_hid,
			'hid_to_class' : hid_to_class}
		  

	def model_to_theta(self, model):
		"""Convert matrices of model parameters into vector
		
		Args:
			model (dict): dictionary of model parameter matrices
		
		Returns:
			vector: vector of model parameters
		"""
		input_to_hid_transpose = model['input_to_hid'].flatten()
		hid_to_class_transpose = model['hid_to_class'].flatten()
		return np.concatenate(
			(input_to_hid_transpose, hid_to_class_transpose)
		)

	def theta_to_model(self, theta):
		"""Convet vector of model parameters into matices
		
		Args:
			theta (vector): vector of model parameters
		
		Returns:
			dict: dictionary of model parameter matrices
		"""
		n_hid = theta.size // (self.n_in + self.n_out)
		input_to_hid = theta[:self.n_in*n_hid].reshape(n_hid, self.n_in)
		hid_to_class = theta[self.n_in*n_hid : theta.size].reshape(self.n_out, n_hid)
		return {'input_to_hid': input_to_hid,
			'hid_to_class' : hid_to_class}
	
	def initial_model(self, n_hid):
		"""Set initial parameters
		
		Args:
			n_hid (int): number of hidden units
		
		Returns:
			dict: dictionary of model parameter matrices
		
		Fixed initialization to get reproducible results
		"""
		n_params = (self.n_in + self.n_out) * n_hid
		as_row_vector = np.cos(np.arange(n_params))
		return self.theta_to_model(as_row_vector * 0.1)
	
	def classification_performance(self, model, data):
		"""Determine incorrectly classified cases
		
		Args:
			model (dict): dictionary of layer weight matrices
			data (dict): training inputs		
		Returns:
			float: fraction of data cases that is
				incorrectly classified by the model
		"""
		hid_input = model['input_to_hid'].dot(data['inputs'])
		hid_output = NetPy().logistic(hid_input)
		class_input = model['hid_to_class'].dot(hid_output)

		# max index in col is the chosen class, plus 1
		choices = np.argmax(class_input, axis=0);
		# max index in targets is integer: the target class, plus 1.
		targets = np.argmax(data['targets'], axis=0)
		
		# get mean of incorret values
		return np.mean((choices != targets).astype(int))
	
	def test(self):
		"""Gash test method"""
		for weight_matrix, value in self.model.items():
			print('Matrix: {}, shape: {}, values {}'.format(
				weight_matrix, value.shape, value))
		theta = self.model_to_theta(self.model)
		print(theta)
		model = self.theta_to_model(theta)
		for weight_matrix, value in self.model.items():
			print('Matrix: {}, shape: {}, values {}'.format(
				weight_matrix, value.shape, value))
