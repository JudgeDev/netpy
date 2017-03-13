"""Neural net class in python

Notation - mathematics:
Use space in math expressions: x^2 + 3 x - 17
Superscripts: 2^6, e^3, (3 x)^2, x^(1 / 2)
Subscripts with braces: lim_{x --> 0} sin (x) / x
Division: (x - 6) / (x + 9), 1 / (3 x)
Multiplication when ambiguity is possible: 4 * 3
Function evaluations: sin (x)
Standard abbreviations: sqrt (x), exp (x)
Infinity: infinity or \infty (TeX code)
Limits: lim_{x --> 3^+} 1 / (x - 3)
Derivatives: f' (x), (d/dt) (a t^2 + b t)
Integrals: \int_a^b f (x) dx

Notation - neural networks:
z: input / logit
y: output / activation
w: weight
i: input
D: number of input dimensions
h: hidden layer
D^i_h: number of hidden units in i-th layer
o: ouput layer
L: number of labels
d: data case, m is # of data cases
t: target, T (=D) is # of targets

theta = (W,b) = Wb: set of all parameters for a given model

f_Wb(x) or f(x): classification function associated with a model (Wb/theta)
	P(Y|x,Wb), defined as argmax_kP(Y=k|x,Wb)
L(Wb, D): log-likelihood D of the model defined by parameters Wb/theta
l(Wb, D): empiracal loss of the prediction function f,
	parameterized by Wb on data set D
NLL: negative log-likelihood

floats [0,1] = {x e R | 0 <= x <= 1}
integers [0 .. 9] = {0, 1 .. 9}


Questions:
How to represent layer - superscript/subscript?
- superscript
How to represent indices - superscript/subscript?
- subscript
Is weight in layer below or above?
- below
What is order of indices to get easy vectorisation?
- layer above, layer below
What is dimension of vectors to get easy vectorisation?
- column (or row with transpose) for pre-multiplication by weights

"""

# Python imports

import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# some plotting functions
from matplotlib.colors import colorConverter, ListedColormap 
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from matplotlib import cm # Colormaps
# Allow matplotlib to plot inside this notebook
#%matplotlib inline

class NetPy:
	def __init__(self):
		# Set the seed of the numpy random number generator so that
		# the tutorial is reproducable
		np.random.seed(seed=1)
	
	def draw_graph(self, title, labels, series,
		axes=[None, None, None, None], grid=False, legend=False):
		"""
		General graph drawing function.
		
		axes: [xmin, xmax, ymin, ymax]
		labels: [xlabel, ylabel]
		series: list of tuples (x, y, markers, label)
		markers: 'r' = red, 'g' = green, 'b' = blue, 'c' = cyan,
				 'm' = magenta, 'y' = yellow, 'k' = black, 'w' = white
				 '-' = solid, '--' = dashed, ':' = dotted,
				 '-.' = dot-dashed, '.' = points, 'o' = filled circles,
				 '^' = filled triangles
		You can use a subset TeX markup in any matplotlib text string
		by placing it inside a pair of dollar signs ($)
		"""
		plt.clf()
		# for each line is series
		for line in series:
			plt.plot(line[0], line[1], line[2], label=line[3])
		plt.xlabel(labels[0], fontsize=15)
		plt.ylabel(labels[1], fontsize=15)
		plt.axis(axes)
		plt.title(title)
		if grid: plt.grid()
		if legend: plt.legend(loc=2)
		plt.show()

	def logistic(self, z):
		"""
		Logistic classification function.
		
		If we want to do classification with neural networks we want to
		output a probability distribution over the classes from
		the output targets t.
		For the classification of 2 classes t=1 or t=0 we can use
		the logistic function used in logistic regression.
		For multiclass classification there exists an extension of
		this logistic function called the softmax function which is
		used in multinomial logistic regression.
		
		Logistic function
		The goal is to predict the target class t from an input z.
		The probability P(t=1|z) that input z is classified as
		class t=1 is represented by the output y of
		the logistic function computed as y = σ(z) defined as:
			σ(z) = 1/(1+e^−z)
		This logistic function maps the input z to
		an output between 0 and 1.
		We can write the probabilities that the class is t=1 or t=0
		given input z as:
			P(t=1|z) = σ(z) = 1/(1+e^−z)
			P(t=0|z) = 1−σ(z) = e^−z/(1+e^−z)
		
		Note that input z to the logistic function corresponds to
		the log odds ratio of P(t=1|z) over P(t=0|z):
			logP(t=1|z)/P(t=0|z) = log(1/(1+e^−z)/e^−z/(1+e^−z))
			= log(1/e^−z) = log(1) − log(e^−z) = z
		
		This means that the log odds ratio log(P(t=1|z)/P(t=0|z))
		changes linearly with z.
		And if z = x*w as in neural networks, this means that
		the log odds ratio changes linearly with the parameters
		w and input samples x.
		"""
		return 1 / (1 + np.exp(-z))

	def dSigma(self, z):
		"""
		Derivative of the logistic function.
		
		Since neural networks typically use gradient based
		opimization techniques such as gradient descent,
		it is important to define the derivative of the output y of
		the logistic function with respect to its input z.
		∂y/∂z can be calculated as:
			∂y/∂z = ∂σ(z)/∂z = ∂(1/(1+e−^z))/∂z = −1/(1+e^−z)^2*e^−z*−1
			= 1/(1+e^−z).e^−z/(1+e^−z)
		and since 1−σ(z)) = 1−1/(1+e^−z) = e^−z/(1+e^−z),
		this can be rewritten as:
			∂y/∂z = 1/(1+e^−z).e^−z/(1+e^−z)
			= σ(z)*(1−σ(z)) = y(1−y)
		"""
		return self.logistic(z) * (1 - self.logistic(z))
	
	def cost_sq(self, y, t):
		"""
		The squared error cost function.
		
		The squared error cost is defined as ξ = ∑_{i=1,N}(t_i−y_i)^2,
		with N the number of samples in the training set.
		The optimization goal is thus: argmin_w∑_{i=1,N}(t_i−y_i)^2
		Notice that we take the sum of errors over all samples,
		which is known as batch training. We could also update
		the parameters based upon one sample at a time,
		which is known as online training. 
		This cost function for variable w is plotted in
		the figure below.
		The value w=2 is at the minimum of the cost function
		(bottom of the parabola), this value is the same value as
		the slope we choose for f(x). Notice that this function is
		convex and that there is only one minimum: the global minimum.
		While every squared error cost function for linear regression
		is convex, this is not the case for other models
		and other cost functions.
		"""
		return ((t - y)**2).sum()		

	def cost_ce(self, y, t):
		"""
		Cross-entropy cost function for the logistic function.
		
		The output of the model y=σ(z) can be interpreted as
		a probability y that input z belongs to one class (t=1),
		or probability 1−y that z belongs to the other class (t=0)
		in a two class classification problem.
		We note this down as: P(t=1|z)=σ(z)=y.
		
		The neural network model will be optimized by maximizing
		the likelihood that a given set of parameters θ of the model
		can result in a prediction of the correct class of each
		input sample. The parameters θ transform each input sample i
		into an input to the logistic function z_i.
		The likelihood maximization can be written as:
			argmax_θL(θ|t,z) = argmax_θ∏_{i=1,n}L(θ|t_i,z_i)
		
		The likelihood L(θ|t,z) can be rewritten as
		the joint probability of generating t and z given
		the parameters θ: P(t,z|θ).
		Since P(A,B) = P(A|B)*P(B) this can be written as:
			P(t,z|θ) = P(t|z,θ)P(z|θ)
		
		Since we are not interested in the probability of z,
		we can reduce this to:
			L(θ|t,z) = P(t|z,θ) = ∏_{i=1,n}P(t_i|z_i,θ).
			
		Since t_i is a Bernoulli variable,
		and the probability P(t|z)=y is fixed for
		a given θ we can rewrite this as:
			P(t|z) = ∏_{i=1,n}P(t_i=1|z_i)^t_i * (1−P(t_i=1|z_i))^(1−t_i)
			= ∏_{i=1,n}y_i_ti * (1−y_i)^(1−t_i)
		
		Since the logarithmic function is a monotone increasing function,
		we can optimize the log-likelihood function argmax_θlogL(θ|t,z)
		This maximum will be the same as the maximum from
		the regular likelihood function.
		The log-likelihood function can be written as:
			logL(θ|t,z) = log∏_{i=1,n}y_i^t_i * (1−y_i)^(1−t_i)
			= ∑_{i=1,n}t_i.log(y_i) + (1−t_i).log(1−y_i)
		
		Minimizing the negative of this function
		(minimizing the negative log likelihood) corresponds to
		maximizing the likelihood.
		This error function ξ(t,y) is typically known as
		the cross-entropy error function (also known as log-loss):
			ξ(t,y) = −logL(θ|t,z)
			= −∑_{i=1,n}[t_i.log(y_i) + (1−t_i).log(1−y_i)]
			= −∑_{i=1,n}[t_ilog(σ(z) + (1−t_i)log(1−σ(z))]
		
		This function looks complicated but besides
		the previous derivation there are a couple of intuitions why
		this function is used as a cost function for logistic regression.
		First of all it can be rewritten as:
			ξ(t_i,y_i )= −log(y_i) 		if t_i=1
						−log(1−y_i) 	if t_i=0
		
		Which in the case of t_i=1 is 0 if y_i=1 (−log(1)=0)
		and goes to infinity as yi→0 (limy→0−log(y)=+∞).
		The reverse effect is happening if t_i=0.
		 
		So what we end up with is a cost function that is 0 if
		the probability to predict the correct class is 1 and
		goes to infinity as the probability to predict
		the correct class goes to 0.
		 
		Notice that the cost function ξ(t,y) is equal to
		the negative log probability that z is classified as
		its correct class: 
			−log(P(t=1|z)) = −log(y), 
			−log(P(t=0|z)) = −log(1−y). 
		
		By minimizing the negative log probability, we will maximize
		the log probability. And since t can only be 0 or 1,
		we can write ξ(t,y) as:
			ξ(t,y) = −t*log(y) − (1−t)*log(1−y)
		
		Which will give:
			ξ(t,y) = −∑_{i=1,n}[t_i.log(y_i) + (1−t_i).log(1−y_i)]
		if we sum over all n samples.
		
		Another reason to use the cross-entropy function is that
		in simple logistic regression this results in
		a convex cost function, of which the global minimum will be
		easy to find. Note that this is not necessarily
		the case anymore in multilayer neural networks.
		
		@TODO - move to separate function?
		Derivative of the cross-entropy cost function for
		the logistic function
		
		The derivative ∂ξ/∂y of the cost function with respect to
		its input can be calculated as:
			∂ξ/∂y = ∂/∂y(−t*log(y) − (1−t)*log(1−y))
			= ∂/∂y(−t*log(y)) + ∂/∂y(−(1−t)*log(1−y))
			= −t/y + (1−t)/(1−y) = (y−t)/y(1−y)
		
		This derivative will give a nice formula if it is used to
		calculate the derivative of the cost function with respect to
		the inputs of the classifier ∂ξ/∂z since the derivative of
		the logistic function is ∂y/∂z=y(1−y):
			∂ξ/∂z = ∂y/∂z.∂ξ/∂y = y(1−y).(y−t)/y(1−y) = y−t
		"""
		return - np.sum(np.multiply(t, np.log(y))
				+ np.multiply((1-t), np.log(1-y)))			
	
	def linear_regression(self):
		"""
		Linear regression using a very simple neural network.
		
		@TODO take out data specific (and model definition) parts and put in their own class
		
		The simplest neural network possible:
		a 1 input 1 output linear regression model that has
		the goal to predict the target value t from the input value x.
		The network is defined as having an input x which gets
		transformed by the weight w to generate the output y by
		the formula y = x*w, and where y needs to approximate
		the targets t as good as possible as defined by a cost function.
		
		We will approximate the targets t with the outputs of
		the model y by minimizing the squared error cost function
		(= squared Euclidian distance).
		The squared error cost function is defined as (t−y)^2.
		The minimization of the cost will be done with
		the gradient descent optimization algorithm which is
		typically used in training of neural networks.		
		
		
		This network can be represented graphically as:
		
		X ------ W -------> Y
		
		In regular neural networks, we typically have multiple layers,
		non-linear activation functions, and a bias for each node.
		Here we only have one layer with one weight parameter w,
		no activation function on the output, and no bias.
		In simple linear regression the parameter w and bias are
		typically combined into the parameter vector β where bias is
		the y-intercept and w is the slope of the regression line.
		In linear regression, these parameters are typically fitted via
		the least squares method.
		
		The neural network model is implemented in
		the nn(x, w) function,
		where '101l' means "one input, no hidden, one linear output"
		"""
		def nn101l(x, w): return x * w
		
		"""
		Define the target function.
		
		In this example, the targets t will be generated from a function f
		and additive gaussian noise sampled from N(0,0.2), where N
		is the normal distribution with mean 0 and variance 0.2.
		f is defined as f(x) = x*2, with x the input samples, slope 2
		and intercept 0.
		t is f(x) + N(0,0.2)
		We will sample 20 input samples x from the uniform distribution
		between 0 and 1, and then generate the target output values t
		by the process described above. These resulting inputs x
		and targets t are plotted against each other in the figure
		together with the original f(x) line without the gaussian noise.
		Note that x is a vector of individual input samples x_i,
		and that t is a corresponding vector of target values.
		"""
		print(self.linear_regression.__doc__)		
		# Define the vector of input samples as x, with 20 values
		# sampled from a uniform distribution between 0 and 1
		x = np.random.uniform(0, 1, 20)
		
		# Generate the target values t from x with small gaussian noise
		# so the estimation won't be perfect.
		# Define a function f that represents the line that
		# generates t without noise
		def f(x): return x * 2
		
		# Create the targets t with some gaussian noise
		noise_variance = 0.2  # Variance of the gaussian noise
		# Gaussian noise error for each sample in x
		noise = np.random.randn(x.shape[0]) * noise_variance
		# Create targets t
		t = f(x) + noise
		
		# Plot the targets/x and initial line
		self.draw_graph('inputs (x) vs targets (t)',
			['$x$', '$t$'], 
			# target t vs x
			[(x, t, 'o', 't'),
			# initial line
			([0, 1], [f(0), f(1)], 'b-', 'f(x)')],
			[None, None, 0, 2], True, True
		)

		
		# Plot the cost vs the given weight w
		# Define a vector of weights for which we want to plot the cost
		ws = np.linspace(0, 4, num=100)  # weight values
		# cost for each weight in ws
		# lambda defines function taking w and returning cost
		# vectorize does this for all ws
		cost_ws = np.vectorize(lambda w: self.cost_sq(nn101l(x, w) , t))(ws)
		
		# Plot
		self.draw_graph('cost vs. weight',
			['$w$', '$\\xi$'], 
			# target t vs x
			[(ws, cost_ws, 'r-', '')],
			grid=True, legend=False
		)
		
		"""
		Optimizing the cost function.
		
		We will optimize the model y = x*w by tuning parameter w so that
		the squared error cost along all samples is minimized.
		For a simple cost function like in this example,
		you can see by eye what the optimal weight should be.
		But the error surface can be quite complex or have
		a high dimensionality (each parameter adds a new dimension).
		This is why we use optimization techniques to find
		the minimum of the error function.
		
		Gradient descent
		One optimization algorithm commonly used to train
		neural networks is the gradient descent algorithm.
		The gradient descent algorithm works by taking
		the derivative of the cost function ξ with respect to
		the parameters at a specific position on this cost function,
		and updates the parameters in the direction of
		the negative gradient.
		The parameter w is iteratively updated by taking steps
		proportional to the negative of the gradient:
			w(k+1) = w(k)−Δw(k)
		with w(k) the value of w at iteration k during
		the gradient descent. 
		
		Δw is defined as: 
			Δw = μ∂ξ/∂w
		with μ the learning rate, which is how big of a step you take
		along the gradient, and ∂ξ/∂w the gradient of
		the cost function ξ with respect to the weight w.
		For each sample i this gradient can be split according to
		the chain rule into:
			∂ξ_i/∂w = ∂y_i/∂w.∂ξ_i/∂y_i
		where ξ_i is the squared error cost, so the ∂ξ_i/∂y_i term
		can be written as:
		 ∂ξ_i/∂y_i = ∂(t_i−y_i)^2/∂y_i = −2(t_i−y_i)=2(y_i−t_i)
		
		And since y_i = x_i*w we can write ∂y_i/∂w as:
			∂y_i/∂w = ∂(x_i*w)/∂w = x_i
		
		So the full update function Δw for sample i will become:
			Δw = μ∗∂ξ_i/∂w = μ*2x_i(y_i−t_i)
		
		In the batch processing, we just add up all the gradients
		for each sample:
			Δw = μ*2*∑_{i=1,N}x_i.(y_i−t_i)
		
		To start out the gradient descent algorithm, you typically
		start with picking the initial parameters at random and
		start updating these parameters with Δw until convergence.
		The learning rate needs to be tuned separately as
		a hyperparameter for each neural network.
		"""
		
		# define the gradient function - ∂ξ/∂w
		# Remember that y = nn(x, w) = x * w
		def gradient_sq(w, x, t): 
		    return 2 * x * (nn101l(x, w) - t)
		
		# define the update function Δw
		def delta_w(w_k, x, t, learning_rate):
		    return learning_rate * gradient_sq(w_k, x, t).sum()
		
		# Set the initial weight parameter
		w = 0.1
		# Set the learning rate
		learning_rate = .1
		
		# Start performing the gradient descent updates,
		# and print the weights and cost:
		# number of gradient descent updates
		nb_of_iterations = 4  
		# List to store the weight,costs values
		w_cost = [(w, self.cost_sq(nn101l(x, w), t))] 
		for i in range(nb_of_iterations):
		    # Get the delta w update
		    dw = delta_w(w, x, t, learning_rate)  
		    # Update the current weight parameter
		    w = w - dw  
		    # Add weight,cost to list
		    w_cost.append((w, self.cost_sq(nn101l(x, w), t)))  
		
		# Print the final w, and cost
		for i in range(len(w_cost)):
		    print('finnal weight: w({}): {:.4f} \t cost: {:.4f}'
				.format(i, w_cost[i][0], w_cost[i][1]))
		
		"""
		Plot Gradient descent updates
		
		Shows the gradient descent updates of the weight parameters for
		2 iterations. The blue dots represent the weight parameter
		values w(k) at iteration.
		Notice how the update differs from the position of the weight
		and the gradient at that point. The first update takes
		a much larger step than the second update because
		the gradient at w(0) is much larger than the gradient at w(1).
		"""
		
		plt.clf()
		plt.plot(ws, cost_ws, 'r-')  # Plot the error curve
		# Plot the updates
		# only first two points
		for i in range(len(w_cost)-2):
		    w1, c1 = w_cost[i]
		    w2, c2 = w_cost[i+1]
		    plt.plot(w1, c1, 'bo')
		    plt.plot([w1, w2],[c1, c2], 'b-')
		    plt.text(w1, c1+0.5, '$w({})$'.format(i)) 
		# Show figure
		plt.xlabel('$w$', fontsize=15)
		plt.ylabel('$\\xi$', fontsize=15)
		plt.title('Gradient descent updates plotted on cost function')
		plt.grid()
		plt.show()
		
		"""
		Plot the fitted line against the target line
		
		The regression line fitted by gradient descent is shown in
		the figure below. The fitted line (red) lies close to
		the original line (blue), which is what we tried to
		approximate via the noisy samples.
		Notice that both lines go through point (0,0), this is because
		we didn't have a bias term, which represents the intercept,
		the intercept at x=0 is thus t=0.
		"""
		
		# Plot the target t versus the input x
		self.draw_graph('inputs (x) vs targets (t)',
			['$input x$', '$targe t$'], 
			# target t vs x
			[(x, t, 'o', 't'),
			# initial line
			([0, 1], [0*w, 1*w], 'r-', 'fitted line')],
			[None, None, 0, 2], True, True
		)

	def logistic_regression(self):
		"""
		Logistic regression (classification)
		
		While the previous tutorial described a very simple
		one-input-one-output linear regression model,
		this tutorial will describe a 2-class classification
		neural network with two input dimensions.
		This model is known in statistics as the logistic regression
		model. This network can be represented graphically as:
		
		x_1 --- w_1 ---
		               |
		               |--------> y
		               |
		x_2 --- w_2 ---
		
		"""
		
		"""
		Define the class distributions.
		
		In this example the target classes t will be generated from
		2 class distributions: blue (t=1) and red (t=0).
		Samples from both classes are sampled from
		their respective distributions. These samples are plotted in
		the figure below.
		Note that X is a N×2 matrix of individual input samples x_i,
		and that t is a corresponding N×1 vector of target values t_i.
		"""

		print(self.logistic_regression.__doc__)	
		
		# Define and generate the samples
		# The number of sample in each class
		nb_of_samples_per_class = 20  
		red_mean = [-1,0]  # The mean of the red class
		blue_mean = [1,0]  # The mean of the blue class
		std_dev = 1.2  # standard deviation of both classes
		# Generate samples from both classes
		x_red = (np.random.randn(nb_of_samples_per_class, 2)
			* std_dev + red_mean)
		x_blue = (np.random.randn(nb_of_samples_per_class, 2)
			* std_dev + blue_mean)
		
		# Merge samples in set of input variables x,
		# and corresponding set of output variables t
		X = np.vstack((x_red, x_blue))
		t = np.vstack((np.zeros((nb_of_samples_per_class,1)),
			np.ones((nb_of_samples_per_class,1))))
		
		# Plot both classes on the x1, x2 plane
		self.draw_graph('red vs. blue classes in the input space',
			['$x_1$', '$x_2$'], 
			# red class
			[(x_red[:,0], x_red[:,1], 'ro', 'class red'),
			# blue class
			(x_blue[:,0], x_blue[:,1], 'bo', 'class blue')],
			[-4, 4, -4, 4], True, True
		)
		
		"""
		Prediction and cost functions.
		
		The goal is to predict the target class t from
		the input values x. The network is defined as having
		an input x=[x_1,x_2] which gets transformed by
		the weights w=[w_1,w_2] to generate the probability that
		sample x belongs to class t=1.
		This probability P(t=1|x,w) is represented by the output y of
		the network computed as y=σ(x*w^T)
		where σ is the logistic function.
		
		The cost function used to optimize the classification is
		the cross-entropy error function.		
		The output of the cost function with respect to
		the parameters w over all samples x is plotted in the code below.
		
		The neural network output is implemented by the nn(x, w) method,
		and the neural network prediction by the nn_predict(x,w) method.
		"""
		# Plot the logistic function
		z = np.linspace(-6,6,100)
		self.draw_graph('logistic function',
			['$z$', '$\sigma(z)$'], 
			[(z, self.logistic(z), 'b-', '')],
			grid=True, legend=False
		)
		
		# Plot the derivative of the logistic function
		self.draw_graph('derivative of the logistic function',
			['$z$', '$\\frac{\\partial \\sigma(z)}{\\partial z}$'], 
			[(z, self.dSigma(z), 'r-', '')],
			grid=True, legend=False
		)
		
		# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
		# 'iol' = input, logistic output
		def nniol(x, w):
			return self.logistic(x.dot(w.T))
		
		# Define the neural network prediction function that
		# only returns 1 or 0 depending on the predicted class
		def nn_predict(x,w):
			# round to default zero decimals
			return np.around(nniol(x,w))
		
		# Define a vector of weights for which we want to plot the cost
		plt.clf()
		# compute the cost nb_of_ws times in each dimension
		nb_of_ws = 100 
		ws1 = np.linspace(-5, 5, num=nb_of_ws) # weight 1
		ws2 = np.linspace(-5, 5, num=nb_of_ws) # weight 2
		ws_x, ws_y = np.meshgrid(ws1, ws2) # generate grid
		# initialize cost matrix
		cost_ws = np.zeros((nb_of_ws, nb_of_ws)) 
		# Fill the cost matrix for each combination of weights
		for i in range(nb_of_ws):
			for j in range(nb_of_ws):
				cost_ws[i,j] = self.cost_ce(nniol(X, np.asmatrix(
					[ws_x[i,j], ws_y[i,j]])) , t)
		
		# Plot the cost function surface
		plt.contourf(ws_x, ws_y, cost_ws, 20, cmap=cm.pink)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('$\\xi$', fontsize=15)
		plt.xlabel('$w_1$', fontsize=15)
		plt.ylabel('$w_2$', fontsize=15)
		plt.title('Cost function surface')
		plt.grid()
		plt.show()
		
		"""
		Gradient descent optimization of the cost function.
		
		The gradient descent algorithm works by taking the derivative
		of the cost function ξ with respect to the parameters,
		and updates the parameters in the direction of
		the negative gradient.
		
		The parameters w are updated by taking steps proportional to
		the negative of the gradient:
			w(k+1) = w(k) − Δw(k+1)
		Δw is defined as:
			Δw = μ.∂ξ/∂w with μ the learning rate.
		
		∂ξ_i/∂w, for each sample i is computed as follows:
			∂ξ_i/∂w = ∂z_i/∂w.∂y_i/∂z_i.∂ξ_i/∂y_i
		where y_i=σ(z_i) is the output of the logistic neuron,
		and z_i=x_i*w^T the input to the logistic neuron.
		
		∂ξ_i/∂y_i can be calculated as:
			∂ξ_i/∂y_i = y_i−t_i/y_i(1−y_i)
		
		∂yi/∂zi can be calculated as:
			∂y_i/∂z_i = y_i(1−y_i)
		
		zi/∂w can be calculated as:
			∂z/∂w = ∂(x*w)∂w=x
		
		Bringing this together we can write:
		
			∂ξ_i/∂w = ∂z_i/∂w.∂y_i/∂z_i.∂ξ_i/∂y_i
			= x*y_i(1−y_i)*y_i−t_i/(y_i(1−y_i)) = x*(y_i−t_i)
		
		Notice how this gradient is the same (negating
		the constant factor) as the gradient of
		the squared error regression.
		
		So the full update function Δw_j for each weight will become:
			Δw_j = μ*∂ξ_i/∂w_j = μ*x_j*(y_i−t_i)
		
		In the batch processing, we just add up all the gradients
		for each sample:
			Δw_j = μ*∑_{i=1,N}x_i,j(y_i−t_i)
		
		The gradient descent algorithm typically starts by picking
		the initial parameters at random and updates these parameters
		according to the delta rule with Δw until convergence.
		"""
		
		# define the gradient function ∂ξ/∂w
		def gradient_ce(w, x, t): 
		    return (nniol(x, w) - t).T * x
		
		# define the update function Δw
		# return a value for each weight in a vector
		def delta_w_ce(w_k, x, t, learning_rate):
		    return learning_rate * gradient_ce(w_k, x, t)
		# ?? why no .sum here

		"""
		Gradient descent updates.
		
		Gradient descent is run on the example inputs X and targets t
		for 10 iterations. The first 3 iterations are shown in
		the plotted figure. The blue dots represent
		the weight parameter values w(k) at iteration k.
		"""
		
		# Set the initial weight parameter
		w = np.asmatrix([-4, -2])
		# Set the learning rate
		learning_rate = 0.05
		
		# Start the gradient descent updates and plot the iterations
		# Number of gradient descent updates
		nb_of_iterations = 10  
		# List to store the weight values over the iterations
		w_iter = [w]  
		for i in range(nb_of_iterations):
		    # Get the delta w update
		    dw = delta_w_ce(w, X, t, learning_rate)  
		    w = w-dw  # Update the weights
		    w_iter.append(w)  # Store the weights for plotting
		
		# Plot the first weight updates on the error surface
		# Plot the error surface
		plt.clf()
		plt.contourf(ws_x, ws_y, cost_ws, 20, alpha=0.9, cmap=cm.pink)
		cbar = plt.colorbar()
		cbar.ax.set_ylabel('cost')
		
		# Plot the updates
		for i in range(1, 4): 
		    w1 = w_iter[i-1]
		    w2 = w_iter[i]
		    # Plot the weight cost value
		    plt.plot(w1[0,0], w1[0,1], 'bo')  
		    # and the line that represents the update
		    plt.plot([w1[0,0], w2[0,0]], [w1[0,1], w2[0,1]], 'b-')
		    plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'
				.format(i), color='b')
		w1 = w_iter[3]  
		# Plot the last weight
		plt.plot(w1[0,0], w1[0,1], 'bo')
		plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'
			.format(4), color='b') 
		# Show figure
		plt.xlabel('$w_1$', fontsize=15)
		plt.ylabel('$w_2$', fontsize=15)
		plt.title('Gradient descent updates on cost surface')
		plt.grid()
		plt.show()
		
		"""
		Visualization of the trained classifier.
		
		The resulting decision boundary of running gradient descent on
		the example inputs X and targets t is shown in the next plot.
		The background color refers to the classification decision of
		the trained classifier.
		
		Note that since this decision plane is linear that not all
		examples can be classified correctly. Two blue dots will be 
		misclassified as red, and four red spots will be
		misclassified as blue.
		
		Note that the decision boundary goes through the point (0,0)
		since we don't have a bias parameter on the logistic output unit.
		"""
		
		# Plot the resulting decision boundary
		# Generate a grid over the input space to plot the color of the
		#  classification at that grid point
		plt.clf()
		nb_of_xs = 200
		xs1 = np.linspace(-4, 4, num=nb_of_xs)
		xs2 = np.linspace(-4, 4, num=nb_of_xs)
		xx, yy = np.meshgrid(xs1, xs2) # create the grid
		# Initialize and fill the classification plane
		classification_plane = np.zeros((nb_of_xs, nb_of_xs))
		for i in range(nb_of_xs):
		    for j in range(nb_of_xs):
		        classification_plane[i,j] = nn_predict(np.asmatrix(
					[xx[i,j], yy[i,j]]) , w)
		# Create a color map to show the classification colors
		# of each grid point
		cmap = ListedColormap([
		        colorConverter.to_rgba('r', alpha=0.30),
		        colorConverter.to_rgba('b', alpha=0.30)])
		
		
		# Plot the classification plane with decision boundary
		# and input samples
		plt.contourf(xx, yy, classification_plane, cmap=cmap)
		plt.plot(x_red[:,0], x_red[:,1], 'ro', label='target red')
		plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='target blue')
		plt.grid()
		plt.legend(loc=2)
		plt.xlabel('$x_1$', fontsize=15)
		plt.ylabel('$x_2$', fontsize=15)
		plt.title('red vs. blue classification boundary')
		plt.show()	
	
	def hidden_layer(self):
		"""
		Hidden layer.
		
		While the previous functions represent very simple single layer
		regression and classification models,
		the following will describe a 2-class classification
		neural network with 1 input dimension,
		and a non-linear hidden layer with 1 neuron.
		This network can be represented graphically as:
			
		x --- w_h --- h --- w_o --- y
		
		"""
		
		"""
		Define the dataset.
		
		In this example the target classes t corresponding to
		the inputs will be generated from 2 class distributions:
			blue (t=1) and red (t=0).
		Where the red class is a multimodal distribution that
		surrounds the distribution of the blue class.
		This results in a 1D dataset that is not linearly separable.
		These samples are plotted by the code below.
		
		The previous model won't be able to classify both classes
		correctly since it can learn only linear separators.
		By adding a hidden layer with a non-linear transfer function,
		the model will be able to train a non-linear classifier.
		"""
		print(self.hidden_layer.__doc__)
		# Define and generate the samples
		nb_of_samples_per_class = 20  # The number of sample in each class
		blue_mean = [0]  # The mean of the blue class
		red_left_mean = [-2]  # The mean of the red class
		red_right_mean = [2]  # The mean of the red class
		
		std_dev = 0.5  # standard deviation of both classes
		# Generate samples from both classes
		x_blue = (np.random.randn(nb_of_samples_per_class, 1)
			* std_dev + blue_mean)
		x_red_left = (np.random.randn(nb_of_samples_per_class//2, 1)
			* std_dev + red_left_mean)
		x_red_right = (np.random.randn(nb_of_samples_per_class//2, 1)	
		 * std_dev + red_right_mean)
		
		# Merge samples in set of input variables x
		# merge corresponding set of output variables t
		# blue = 1, red = 0
		x = np.vstack((x_blue, x_red_left, x_red_right))
		t = np.vstack((np.ones((x_blue.shape[0],1)), 
		               np.zeros((x_red_left.shape[0],1)), 
		               np.zeros((x_red_right.shape[0], 1))))
		
		# Plot samples from both classes as lines on a 1D space
		##plt.clf()
		# figsize tuple w,h in inches
		plt.figure(figsize=(8, 2))
		plt.xlim(-3,3)
		plt.ylim(-1,1)
		# Plot samples
		# zeros array same shape as x_
		# markersize ms in points
		plt.plot(x_blue, np.zeros_like(x_blue), 'b|', ms = 120) 
		plt.plot(x_red_left, np.zeros_like(x_red_left), 'r|', ms = 120) 
		# get current axes gca and hide y axis
		plt.plot(x_red_right, np.zeros_like(x_red_right), 'r|', ms = 120) 
		plt.gca().axes.get_yaxis().set_visible(False)
		plt.title('Input samples from the blue and red class')
		plt.xlabel('$x$', fontsize=15)
		plt.show()
		
		"""
		Non-linear transfer function.
		
		The non-linear transfer function used in the hidden layer of
		this example is the Gaussian radial basis function (RBF). 
		The RBF is a transfer function that is not usually used
		in neural networks, except for radial basis function networks.
		One of the most common transfer functions in neural networks
		is the sigmoid transfer function.
		The RBF will allow to separate the blue samples from
		the red samples in this simple example by only activating for
		a certain region around the origin.
		The RBF is plotted in the figure below and is defined
		in this example as:
		  	RBF=ϕ(z) = e^−z^2
		
		The derivative of this RBF function is:
			dϕ(z)/dz = −2ze^−z^2 = −2zϕ(z)
		"""
		
		def rbf(z):
			return np.exp(-z**2)	
				
		# Plot the rbf function
		plt.close()
		z = np.linspace(-6,6,100)
		plt.plot(z, rbf(z), 'b-')
		plt.xlabel('$z$', fontsize=15)
		plt.ylabel('$e^{-z^2}$', fontsize=15)
		plt.title('RBF function')
		plt.grid()
		plt.show()

		"""
		Optimization by backpropagation.
		
		We will train this model by using the backpropagation algorithm
		that is typically used to train neural networks.
		Each iteration of the backpropagation algorithm consists of
		two steps:
		
		1. A forward propagation step to compute
		the output of the network.
		2. A backward propagation step in which the error at
		the end of the network is propagated backward through all
		the neurons while updating their parameters.
		
		1. Forward step
		During the forward step, the input will be propagated
		layer by layer through the network to compute
		the final output of the network.
		
		Compute activations of hidden layer
		The activations h of the hidden layer will be computed by:
			h = ϕ(x*w_h) = e^−(x*w_h)^2
		with w_h the weight parameter that transforms the input
		before applying the RBF transfer function.
		"""
		def hidden_activations(x, wh):
			return rbf(x * wh)
		
		"""
		Compute activations of output
		The output of the final layer and network will be computed by
		passing the hidden activations h as input to
		the logistic output function:
			y = σ(h*w_o−1) = 1/(1+e^(−h*w_o−1))
		with w_o the weight parameter of the output layer.
		Note that we add a bias (intercept) term of −1 to the input of
		the logistic output neuron.
		Remember that the logistic output neuron without bias can only
		learn a decision boundary that goes through the origin (0).
		
		Since the RBF in the hidden layer projects all
		input variables to a range between 0 and +∞,
		the output layer without an intercept will not be able to
		learn any useful classifier, because none of
		the samples will be below 0 and thus lie on the left side of
		the decision boundary.
		By adding a bias term the decision boundary is moved from
		the intercept. Normally the value of this bias termed is
		learned together with the rest of the weight parameters,
		but to keep this model simple we just make this bias constant
		in this example.
		"""
		def output_activations(h , wo):
			return self.logistic(h * wo - 1)
		
		# Define the neural network function
		def nn(x, wh, wo): 
			return output_activations(hidden_activations(x, wh), wo)
		
		# Define the neural network prediction function that only
		# returns 1 or 0 depending on the predicted class
		def nn_predict(x, wh, wo):
			# round to default zero decimals
			return np.around(nn(x, wh, wo))
		
		"""
		2. Backward step
		The backward step will begin with computing the cost at
		the output node. This cost will then be propagated backwards
		layer by layer through the network to update the parameters.
		
		The gradient descent algorithm is used in every layer to
		update the parameters in the direction of the negative gradient.
		
		The parameters w_h and w_o are updated by w(k+1) = w(k)−Δw(k+1).
		Δw is defined as: Δw = μ*∂ξ/∂w with μ the learning rate and
		∂ξ/∂w the gradient of the parameter w with respect to
		the cost function ξ.
		
		Compute the cost function
		The cost function ξ used in this model is
		the cross-entropy cost function:
			ξ(t_i,y_i) = −[t_i.log(y_i) + (1−t_i).log(1−y_i)]
		"""
		def cost_for_param(x, wh, wo, t):
			return self.cost_ce(nn(x, wh, wo) , t)
		
		"""
		This cost function is plotted for the w_h parameters in
		the next figure.
		Note that this error surface is not convex anymore
		and that the w_h parameter mirrors the cost function
		along the w_h=0 axis. 
		Also, notice that this cost function has a very sharp
		gradient around w_h=0 starting from w_o>0 and that
		the minima run along the lower edge of this peak.
		If the learning rate will be to big, the updates might
		jump over the minima gap, onto the sharp gradient.
		Because the gradient is sharp, the update will be large,
		and we might end up further from the minima than we started. 
		"""
		# Define a vector of weights for which we want to plot the cost
		nb_of_ws = 200 # compute the cost nb_of_ws times in each dimension
		wsh = np.linspace(-10, 10, num=nb_of_ws) # hidden weights
		wso = np.linspace(-10, 10, num=nb_of_ws) # output weights
		ws_x, ws_y = np.meshgrid(wsh, wso) # generate grid
		cost_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize cost matrix
		# Fill the cost matrix for each combination of weights
		for i in range(nb_of_ws):
		    for j in range(nb_of_ws):
		        cost_ws[i,j] = self.cost_ce(nn(x, ws_x[i,j], ws_y[i,j]) , t)
		
		# Plot the cost function surface
		plt.clf()
		fig = plt.figure()
		ax = Axes3D(fig)
		# plot the surface
		surf = ax.plot_surface(ws_x, ws_y, cost_ws, linewidth=0, cmap=cm.pink)
		ax.view_init(elev=60, azim=-30)
		cbar = fig.colorbar(surf)
		ax.set_xlabel('$w_h$', fontsize=15)
		ax.set_ylabel('$w_o$', fontsize=15)
		ax.set_zlabel('$\\xi$', fontsize=15)
		cbar.ax.set_ylabel('$\\xi$', fontsize=15)
		plt.title('Cost function surface')
		plt.grid()
		plt.show()
		
		"""		
		Update the output layer.
		
		Using the chain rule, the gradient for sample i at the output
		∂ξ_i/∂w_o is:
			∂ξ_i/∂w_o = ∂z_oi/∂w_o.∂y_i/∂z_oi.∂ξ_i/∂y_i
			= h_i(y_i − t_i) = h_i*δ_oi
		
		with z_oi = h_i*w_o the hidden layer activation of sample i
		and ∂ξ_i/∂z_oi = δ_oi the gradient of the error at
		the output layer of the neural network with respect to
		the input to this layer.
		"""
		# error function, δ_oi
		def gradient_output(y, t):
			return y - t

		# gradient function for the weight parameter at
		# the output layer, ∂ξ_i/∂w_o
		def gradient_weight_out(h, grad_output): 
			return  h * grad_output		
		"""
		Update the hidden layer.
		
		At the hidden layer the gradient for sample i, ∂ξ_i/∂w_h,
		of the hidden neuron is computed the same way:
			∂ξ_i/∂w_h = ∂z_hi/∂w_h.∂h_i/∂z_hi.∂ξ_i/∂h_i
		with z_hi = x_i*w_h.
		And with ∂ξ_i/∂z_hi = δ_hi the gradient of the error at
		the input of the hidden layer with respect to the input
		to this layer.
		This error can be interpreted as the contribution of z_hi to
		the final error.
		How do we define this error gradient δ_hi at the input of
		the hidden neurons?
		It can be computed as the error gradient propagated back from
		the output layer through the hidden layer:
			δ_hi = ∂ξ_i/∂z_hi = ∂h_i/∂z_hi.∂z_oi/∂h_i.∂ξ_i/∂z_oi
			= (−2*z_hi*h_i)*w_o*(y_i−t_i) = −2*z_hi*h_i*w_o*δ_oi
			
		Thus, and because ∂z_hi/∂w_h = x_i:
			∂ξ_i/∂w_h = x_i.δ_hi
		
		The gradients for each parameter can again be summed up to
		compute the update for a batch of input examples.
		"""
		# gradient function for the hidden layer, δ_h
		def gradient_hidden(wo, grad_output):
			return wo * grad_output

		# gradient function for the weight parameter at
		# the hidden layer, ∂ξ_i/∂w_h
		def gradient_weight_hidden(x, zh, h, grad_hidden):
			return x * -2 * zh * h * grad_hidden
		
		"""
		Backpropagation updates.
		
		To start out the gradient descent algorithm,
		you typically start with picking the initial parameters
		at random and start updating these parameters in
		the direction of the negative gradient with help of
		the backpropagation algorithm.
		One backpropagation iteration is implemented below by
		the backprop_update(x, t, wh, wo, learning_rate) method.
		"""		
		def backprop_update(x, t, wh, wo, learning_rate):
			# Compute the output of the network
			# This can be done with y = nn(x, wh, wo),
			# but we need the intermediate h and zh for
			# the weight updates.
			zh = x * wh
			h = rbf(zh)  # hidden_activations(x, wh)
			y = output_activations(h, wo)
			# Compute the gradient at the output
			grad_output = gradient_output(y, t)
			# Get the delta for wo
			d_wo = learning_rate * gradient_weight_out(h, grad_output)
			# Compute the gradient at the hidden layer
			grad_hidden = gradient_hidden(wo, grad_output)
			# Get the delta for wh
			d_wh = learning_rate * gradient_weight_hidden(
				x, zh, h, grad_hidden)
			# return the update parameters
			return (wh-d_wh.sum(), wo-d_wo.sum())
		
		"""
		 Run backpropagation.
		 
		 An example run of backpropagation for 50 iterations
		 on the example inputs x and targets t is shown in
		 the figure below.
		 The white dots represent the weight parameter values w_h
		 and w_o at iteration k and are plotted on the cost surface.
		 Notice that we decrease the learning rate linearly with
		 each step. This is to make sure that in the end
		 the learning rate is 0 and the sharp gradient will not allow
		 the weight paramaters to fluctuate much during
		 the last few iterations. 
		"""	
		# Set the initial weight parameter
		wh = 2
		wo = -5
		# Set the learning rate
		learning_rate = 0.2

		# Start the gradient descent updates and plot the iterations
		nb_of_iterations = 50  # number of gradient descent updates
		# learning rate update rule
		lr_update = learning_rate / nb_of_iterations 
		# List to store the weight values over the iterations
		w_cost_iter = [(wh, wo, cost_for_param(x, wh, wo, t))]
		for i in range(nb_of_iterations):
			learning_rate -= lr_update # decrease the learning rate
			# Update the weights via backpropagation
			wh, wo = backprop_update(x, t, wh, wo, learning_rate) 
			# Store the values for plotting
			w_cost_iter.append((wh, wo, cost_for_param(x, wh, wo, t)))  

		# Print the final cost
		print('final cost is {:.2f} for weights wh: {:.2f} and wo: {:.2f}'
			.format(cost_for_param(x, wh, wo, t), wh, wo))		
		
		# Plot the weight updates on the error surface
		# Plot the error surface
		plt.clf()
		fig = plt.figure()
		ax = Axes3D(fig)
		surf = ax.plot_surface(
			ws_x, ws_y, cost_ws, linewidth=0, cmap=cm.pink)
		ax.view_init(elev=60, azim=-30)
		cbar = fig.colorbar(surf)
		cbar.ax.set_ylabel('$\\xi$', fontsize=15)

		# Plot the updates
		for i in range(1, len(w_cost_iter)):
			wh1, wo1, c1 = w_cost_iter[i-1]
			wh2, wo2, c2 = w_cost_iter[i]
			# Plot the weight cost value
			ax.plot([wh1], [wo1], [c1], 'w+')  
			# and the line that represents the update 
			ax.plot([wh1, wh2], [wo1, wo2], [c1, c2], 'w-')
		# Plot the last weights
		wh1, wo1, c1 = w_cost_iter[len(w_cost_iter)-1]
		ax.plot([wh1], [wo1], c1, 'w+')
		# Show figure
		ax.set_xlabel('$w_h$', fontsize=15)
		ax.set_ylabel('$w_o$', fontsize=15)
		ax.set_zlabel('$\\xi$', fontsize=15)
		plt.title('Gradient descent updates on cost surface')
		plt.grid()
		plt.show()
		
		"""
		Visualization of the trained classifier.
		
		The resulting decision boundary of running backpropagation on
		the example inputs x and targets t is shown in the figure below.
		The background color (blue, red) refers to
		the classification decision of the trained classifier at
		that position in the input space.
		Note that all examples are classified correctly by
		the trained classifier. 
		"""	
		# Plot the resulting decision boundary
		# Generate a grid over the input space to plot the color of the
		#  classification at that grid point
		plt.clf()
		nb_of_xs = 100
		xs = np.linspace(-3, 3, num=nb_of_xs)
		ys = np.linspace(-1, 1, num=nb_of_xs)
		xx, yy = np.meshgrid(xs, ys) # create the grid
		# Initialize and fill the classification plane
		classification_plane = np.zeros((nb_of_xs, nb_of_xs))
		for i in range(nb_of_xs):
			for j in range(nb_of_xs):
				classification_plane[i,j] = nn_predict(xx[i,j], wh, wo)
		# Create a color map to show the classification colors
		# of each grid point
		cmap = ListedColormap([
				colorConverter.to_rgba('r', alpha=0.25),
				colorConverter.to_rgba('b', alpha=0.25)])

		# Plot the classification plane with decision boundary
		# and input samples
		plt.figure(figsize=(8,0.5))
		plt.contourf(xx, yy, classification_plane, cmap=cmap)
		plt.xlim(-3,3)
		plt.ylim(-1,1)
		# Plot samples from both classes as lines on a 1D space
		plt.plot(x_blue, np.zeros_like(x_blue), 'b|', ms = 30) 
		plt.plot(x_red_left, np.zeros_like(x_red_left), 'r|', ms = 30) 
		plt.plot(x_red_right, np.zeros_like(x_red_right), 'r|', ms = 30) 
		plt.gca().axes.get_yaxis().set_visible(False)
		plt.title('Input samples and their classification')
		plt.xlabel('x')
		plt.show()
		
		"""
		Transformation of the input domain.
		
		How is the neural network able to separate the non-linearly
		seperable classes with a linear logistic classifier at
		the output?
		The key is the hidden layer with the non-linear RBF
		transfer function.
		Note that the RBF transfer function is able to transform
		the samples near the origin (blue class) to a value larger
		than 0, and the samples further from the origin (red samples)
		to a value near 0. This projection is plotted in
		the following figure.
		Note that the red samples are located around 0 to the left,
		and that the blue samples are located more to the right.
		This projection is linearly seperable by
		the logistic classifier in the output layer.
		Also, note that the offset of the peak of
		the Gaussian function we use is 0.
		This means that the Gaussian function is centered around
		the origin, which can be noted in the symmetrical
		decision boundaries around the origin on the previous figure. 
		"""
		# Plot projected samples from both classes as lines on a 1D space
		plt.clf()
		plt.figure(figsize=(8,0.5))
		plt.xlim(-0.01,1)
		plt.ylim(-1,1)
		# Plot projected samples
		plt.plot(hidden_activations(x_blue, wh),
			np.zeros_like(x_blue), 'b|', ms = 30) 
		plt.plot(hidden_activations(x_red_left, wh),
			np.zeros_like(x_red_left), 'r|', ms = 30) 
		plt.plot(hidden_activations(x_red_right, wh),
			np.zeros_like(x_red_right), 'r|', ms = 30) 
		plt.gca().axes.get_yaxis().set_visible(False)
		plt.title('Projection of the input samples by the hidden layer.')
		plt.xlabel('h')
		plt.show()
		
	def softmax(self):
		"""
		Softmax classification function.
		
		The logistic output function can only be used for
		the classification between two target classes t=1 and t=0.
		This logistic function can be generalized to output
		a multiclass categorical probability distribution by
		the softmax function. This softmax function ς takes as input
		a C-dimensional vector z and outputs a C-dimensional vector
		y of real values between 0 and 1.
		This function is a normalized exponential and is defined as:
			y_c = ς(z)_c = e^z_c/∑_{d=1,C}e^z_d    for c = 1…C
		
		The denominator ∑_{d=1,C}e^z_d acts as a regularizer to
		make sure that ∑_{c=1,C}y_c = 1.
		As the output layer of a neural network, the softmax function
		can be represented graphically as a layer with C neurons.
		
		We can write the probabilities that the class is t=c
		for c=1…C given input z as:
			P(t=1|z)		ς(z)_1						e^z_1
			.				.							.
			.			=	.		= 1/∑_{d=1,C}e^z_d	.
			.				.							.
			P(t=C|z)		ς(z)_C						e^z_C
			
		where P(t=c|z) is thus the probability that that
		the class is c given the input z.
		
		These probabilities of the output P(t=1|z) for
		an example system with 2 classes (t=1, t=2)
		and input z=[z1,z2] is shown in the figure below.
		The other probability P(t=2|z) will be complementary.
		"""
		def softmax(z):
		    return np.exp(z) / np.sum(np.exp(z))
		
		# Plot the output in function of the weights
		# Define a vector of weights for which we want to plot the ooutput
		nb_of_zs = 200
		zs = np.linspace(-10, 10, num=nb_of_zs) # input 
		zs_1, zs_2 = np.meshgrid(zs, zs) # generate grid
		y = np.zeros((nb_of_zs, nb_of_zs, 2)) # initialize output
		# Fill the output matrix for each combination of input z's
		for i in range(nb_of_zs):
		    for j in range(nb_of_zs):
		        y[i,j,:] = softmax(np.asarray([zs_1[i,j], zs_2[i,j]]))
		# Plot the cost function surfaces for both classes
		plt.clf()
		fig = plt.figure()
		# Plot the cost function surface for t=1
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(zs_1, zs_2, y[:,:,0],
			linewidth=0, cmap=cm.coolwarm)
		ax.view_init(elev=30, azim=70)
		cbar = fig.colorbar(surf)
		ax.set_xlabel('$z_1$', fontsize=15)
		ax.set_ylabel('$z_2$', fontsize=15)
		ax.set_zlabel('$y_1$', fontsize=15)
		ax.set_title ('$P(t=1|\mathbf{z})$')
		cbar.ax.set_ylabel('$P(t=1|\mathbf{z})$', fontsize=15)
		plt.grid()
		plt.show()


if __name__ == '__main__':
	print('Running pyNet1 as main')
	net1 = pyNet1()
	net1.linear_regression()
	net2 = pyNet1()
	net2.logistic_regression()
	net3 = pyNet1()
	net3.hidden_layer()
