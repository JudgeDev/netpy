import six.moves.cPickle as pickle
import gzip
import os
import math
import numpy as np
import matplotlib.pyplot as plt

class Mnist (object):
	""" Class for manipulating MNIST data
	
	data stored in: 
	self.test_set_x, self.test_set_y,
	self.valid_set_x, self.valid_set_y
	self.train_set_x, self.train_set_y
	each set is a tuple(input, target):
		input: numpy.ndarray of 2 dimensions (a matrix) where each row corresponds to an example with 784(28*28) float values between on and 1
		target: numpy.ndarray of 1 dimension (vector) of labels between 0 and 9 corresponding to the inputs 	
	
	"""
	
	def __init__(self):
		pass
	
	
	def load_data(self, dataset):
		""" Loads the dataset
		
			:type dataset: string
			:param dataset: the path to the dataset (here MNIST)
		"""
		
		# Download the MNIST dataset if it is not present
		# file is last component, dir is everything 
		data_dir, data_file = os.path.split(dataset)
		# check if no dir specified and not in current dir
		if data_dir == "" and not os.path.isfile(dataset):
			# modify path to go via 'data' directory
			new_path = os.path.join(
				os.path.split(__file__)[0],
				#"..", data directory at same level as code
				"data",
				dataset
				)
			# change path if file in data directory or file has given name
			if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
					dataset = new_path
		# download file if has given name and not in path 
		if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
			from six.moves import urllib
			origin = (
				'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
				)
			print('Downloading data from %s' % origin)
			urllib.request.urlretrieve(origin, dataset)
		
		print('... loading data')
		
		# Load the dataset
		with gzip.open(dataset, 'rb') as f:
			try:
				train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
			except:
				train_set, valid_set, test_set = pickle.load(f)
		# each set is a tuple(input, target):
		# input: numpy.ndarray of 2 dimensions (a matrix) where
		# each row corresponds to an example with 784(28*28) float
		# values between on and 1
		# target: numpy.ndarray of 1 dimension (vector) of labels
		# between 0 and 9 corresponding to the inputs
		self.test_set_x, self.test_set_y = test_set
		self.valid_set_x, self.valid_set_y = valid_set
		self.train_set_x, self.train_set_y = train_set
	
	def get_datasets(self):
		"""Get the datasets"""
		return [(self.train_set_x, self.train_set_y),
						(self.valid_set_x, self.valid_set_y),
						(self.test_set_x, self.test_set_y)]
	
	def displayData(self, data, image_width=0):
		"""Display 2D data in a nice grid
		
		:type data: 2D array
		:param data: array of data of digits to be displayed 
		:type image_width: int
		:param image_width: width of each image to be displayed
		:rtype: 
		:return: None
		"""

		# padding between images
		pad = 1		
		#number of images and length of image data
		images, length = data.shape
		
		# set image_width automatically if not passed in
		if image_width == 0:
			image_width = round(math.sqrt(length))
		image_height = round(length / image_width)
		
		# compute number of rows and columns to display
		display_rows = math.floor(math.sqrt(images))
		display_cols = math.ceil(images / display_rows)
		
		print('Displaying digit data\nrows: {}, cols: {}, digit width: {}, digit height: {}'.format(
				display_rows, display_cols,
				image_width, image_height))

		# setup blank display for all images
		display_height = pad + display_rows * (image_height + pad)
		display_width = pad + display_cols * (image_width + pad)
		#print('display array height: {}, width{}'.format(
				#display_height, display_width))
		display_array = - np.ones((display_height, display_width))
		
		#copy each example into a patch on the display array
		curr_ex = 0
		for j in range(display_rows):
			for i in range(display_cols):
				if curr_ex >= images:
					 break
				#get the max value of the patch (about 1)
				max_val = max(abs(data[curr_ex, :]))
				# get start positions of patch in display array
				j_start = pad + j * (image_height + pad)
				i_start = pad + i * (image_width + pad)
				# copy patch into display array
				# reshape only seems to work as .method				
				display_array[j_start:j_start + image_height, i_start:i_start + image_width] = data[curr_ex, :].reshape(image_height, image_width, order='C') / max_val
								
				curr_ex += 1  # next image
		
			if curr_ex >= images:
			 	break
		
		#display images
		plt.close()
		h = plt.imshow(display_array,
			cmap='gray',
			interpolation='bilinear',
			origin='upper'
			)
		
		# do not show axis
		plt.axis('off')

		plt.show()

