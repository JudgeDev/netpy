"""Neural net functions in python."""

# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from matplotlib import cm # Colormaps
# Allow matplotlib to plot inside this notebook
#%matplotlib inline
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)

"""
Linear regression using a very simple neural network.

The simplest neural network possible: a 1 input 1 output linear regression model that has the goal to predict the target value t from the input value x.
The network is defined as having an input x which gets transformed by the weight w to generate the output y by the formula y=x*w, and where y needs to approximate the targets t as good as possible as defined by a cost function. This network can be represented graphically as:

X ------ W -------> Y

In regular neural networks, we typically have multiple layers, non-linear activation functions, and a bias for each node.
Here we only have one layer with one weight parameter w, no activation function on the output, and no bias. In simple linear regression the parameter w and bias are typically combined into the parameter vector β where bias is the y-intercept and w is the slope of the regression line. In linear regression, these parameters are typically fitted via the least squares method.

Here, we will approximate the targets t with the outputs of the model y by minimizing the squared error cost function (= squared Euclidian distance). The squared error cost function is defined as (t−y)^2.
The minimization of the cost will be done with the gradient descent optimization algorithm which is typically used in training of neural networks.
"""

"""
Define the target function.

In this example, the targets t will be generated from a function f
and additive gaussian noise sampled from N(0,0.2), where N
is the normal distribution with mean 0 and variance 0.2.
f is defined as f(x)=x*2, with x the input samples, slope 2
and intercept 0.
t is f(x)+N(0,0.2)
We will sample 20 input samples x from the uniform distribution between 0 and 1, and then generate the target output values t
by the process described above. These resulting inputs x
and targets t are plotted against each other in the figure below together with the original f(x) line without the gaussian noise. Note that x is a vector of individual input samples x_i, and that t
is a corresponding vector of target values 
"""

# Define the vector of input samples as x, with 20 values sampled from a uniform distribution
# between 0 and 1
x = np.random.uniform(0, 1, 20)

# Generate the target values t from x with small gaussian noise so the estimation won't
# be perfect.
# Define a function f that represents the line that generates t without noise
def f(x): return x * 2

# Create the targets t with some gaussian noise
noise_variance = 0.2  # Variance of the gaussian noise
# Gaussian noise error for each sample in x
noise = np.random.randn(x.shape[0]) * noise_variance
# Create targets t
t = f(x) + noise

"""
 'r' = red
 'g' = green
 'b' = blue
 'c' = cyan
 'm' = magenta
 'y' = yellow
 'k' = black
 'w' = white
 
 '-' = solid
 '--' = dashed
 ':' = dotted
 '-.' = dot-dashed
 '.' = points
 'o' = filled circles
 '^' = filled triangles
"""
# Plot the target t versus the input x 
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
plt.xlabel('$x$', fontsize=15)  # in italics
plt.ylabel('$t$', fontsize=15)
plt.ylim([0,2])
plt.title('inputs (x) vs targets (t)')
plt.grid()
plt.legend(loc=2)
plt.show()

"""
Define the cost function.

We will optimize the model y=x*w by tuning parameter w so that the squared error cost along all samples is minimized. The squared error cost is defined as ξ=∑_i=1,N(t_i−y_i)^2, with N the number of samples in the training set. The optimization goal is thus: argmin_w∑_i=1,N(ti−yi)^2
Notice that we take the sum of errors over all samples, which is known as batch training. We could also update the parameters based upon one sample at a time, which is known as online training. 
This cost function for variable w is plotted in the figure below. The value w=2 is at the minimum of the cost function (bottom of the parabola), this value is the same value as the slope we choose for f(x). Notice that this function is convex and that there is only one minimum: the global minimum. While every squared error cost function for linear regression is convex, this is not the case for other models and other cost functions.

The neural network model is implemented in the nn(x, w) function, and the cost function is implemented in the cost(y, t) function.
"""

# Define the neural network function y = x * w
# '101l' "one input, no hidden, one linear output"
def nn101l(x, w): return x * w

# Define the cost function
def cost_sq(y, t): return ((t - y)**2).sum()

# Plot the cost vs the given weight w

# Define a vector of weights for which we want to plot the cost
ws = np.linspace(0, 4, num=100)  # weight values
# cost for each weight in ws
# lambda defines function taking w and returning cost
# vectorize does this for all ws
cost_ws = np.vectorize(lambda w: cost_sq(nn101l(x, w) , t))(ws)

# Plot
plt.clf()  # clear previous plot
plt.plot(ws, cost_ws, 'r-')
plt.xlabel('$w$', fontsize=15)
# You can use a subset TeX markup in any matplotlib text string by placing it inside a pair of dollar signs ($)
plt.ylabel('$\\xi$', fontsize=15)
plt.title('cost vs. weight')
plt.grid()
plt.show()

"""
Optimizing the cost function.

For a simple cost function like in this example, you can see by eye what the optimal weight should be. But the error surface can be quite complex or have a high dimensionality (each parameter adds a new dimension). This is why we use optimization techniques to find the minimum of the error function.

Gradient descent
One optimization algorithm commonly used to train neural networks is the gradient descent algorithm. The gradient descent algorithm works by taking the derivative of the cost function ξ with respect to the parameters at a specific position on this cost function, and updates the parameters in the direction of the negative gradient . The parameter w is iteratively updated by taking steps proportional to the negative of the gradient:
w(k+1)=w(k)−Δw(k)
With w(k) the value of w at iteration k during the gradient descent. 
Δw is defined as: 
Δw=μ∂ξ/∂w
With μ the learning rate, which is how big of a step you take along the gradient, and ∂ξ/∂w the gradient of the cost function ξ with respect to the weight w.
For each sample i this gradient can be splitted according to the chain rule into:
∂ξ_i/∂w=∂y_i/∂w.∂ξ_i/∂y_i
Where ξ_i is the squared error cost, so the ∂ξ_i/∂y_i term can be written as:
∂ξ_i/∂y_i=∂(t_i−y_i)^2/∂y_i=−2(t_i−y_i)=2(y_i−t_i)
And since y_i=x_i*w we can write ∂y_i/∂w as:
∂y_i/∂w=∂(x_i*w)/∂w=x_i

So the full update function Δw for sample i will become:
Δw=μ∗∂ξ_i/∂w=μ*2x_i(y_i−t_i)

In the batch processing, we just add up all the gradients for each sample:
Δw=μ*2*∑{i=1,N}x_i(y_i−t_i)

To start out the gradient descent algorithm, you typically start with picking the initial parameters at random and start updating these parameters with Δw until convergence. The learning rate needs to be tuned separately as a hyperparameter for each neural network.

The gradient ∂ξ/∂w is implemented by the gradient(w, x, t) function. Δw is computed by the delta_w(w_k, x, t, learning_rate) . The loop below performs 4 iterations of gradient descent while printing out the parameter value and current cost.
"""

# define the gradient function. Remember that y = nn(x, w) = x * w
def gradient_sq(w, x, t): 
    return 2 * x * (nn101l(x, w) - t)

# define the update function delta w
def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient_sq(w_k, x, t).sum()

# Set the initial weight parameter
w = 0.1
# Set the learning rate
learning_rate = .1

# Start performing the gradient descent updates, and print the weights and cost:
nb_of_iterations = 4  # number of gradient descent updates
w_cost = [(w, cost_sq(nn101l(x, w), t))] # List to store the weight,costs values
for i in range(nb_of_iterations):
    dw = delta_w(w, x, t, learning_rate)  # Get the delta w update
    w = w - dw  # Update the current weight parameter
    w_cost.append((w, cost_sq(nn101l(x, w), t)))  # Add weight,cost to list

# Print the final w, and cost
for i in range(len(w_cost)):
    print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w_cost[i][0], w_cost[i][1]))

"""
Plot Gradient descent updates

Shows the gradient descent updates of the weight parameters for 2 iterations. The blue dots represent the weight parameter values w(k) at iteration.
Notice how the update differs from the position of the weight and the gradient at that point. The first update takes a much larger step than the second update because the gradient at w(0) is much larger than the gradient at w(1).
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

The regression line fitted by gradient descent is shown in the figure below. The fitted line (red) lies close to the original line (blue), which is what we tried to approximate via the noisy samples. Notice that both lines go through point (0,0), this is because we didn't have a bias term, which represents the intercept, the intercept at x=0 is thus t=0.
"""

# Plot the target t versus the input x
plt.clf()
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs. target')
plt.grid()
plt.legend(loc=2)
plt.show()


"""
Logistic classification function.

If we want to do classification with neural networks we want to output a probability distribution over the classes from the output targets t.
For the classification of 2 classes t=1 or t=0 we can use the logistic function used in logistic regression. For multiclass classification there exists an extension of this logistic function called the softmax function which is used in multinomial logistic regression . The following section will explain the logistic function and how to optimize it, the next intermezzo will explain the softmax function and how to derive it.
"""

"""
Logistic function
The goal is to predict the target class t from an input z.
The probability P(t=1|z) that input z is classified as class t=1 is represented by the output y of the logistic function computed as y=σ(z) is the logistic function and is defined as:
σ(z)=1/1+e^−z
This logistic function, implemented below as logistic(z) , maps the input z to an output between 0 and 1

We can write the probabilities that the class is t=1 or t=0 given input z as:
P(t=1|z)=σ(z)=1/1+e^−z
P(t=0|z)=1−σ(z)=e^−z/1+e^−z

Note that input z to the logistic function corresponds to the log odds ratio of P(t=1|z) over P(t=0|z).
logP(t=1|z)/P(t=0|z)=log(1/1+e^−z)/e−^z/1+e^−z)
=log(1/e^−z)=log(1)−log(e^−z)=z

This means that the logg odds ratio log(P(t=1|z)/P(t=0|z)) changes linearly with z. And if z=x*w as in neural networks, this means that the logg odds ratio changes linearly with the parameters w and input samples x.
"""

# Define the logistic function
def logistic(z):
    return 1 / (1 + np.exp(-z))

# Plot the logistic function
plt.clf()
z = np.linspace(-6,6,100)
plt.plot(z, logistic(z), 'b-')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$\sigma(z)$', fontsize=15)
plt.title('logistic function')
plt.grid()
plt.show()

"""
Derivative of the logistic function.

Since neural networks typically use gradient based opimization techniques such as gradient descent it is important to define the derivative of the output y of the logistic function with respect to its input z.
∂y/∂z can be calculated as:
∂y/∂z=∂σ(z)/∂z=∂(1/1+e−^z)/∂z=−1/(1+e^−z)^2*e^−z*−1
=1/1+e^−z.e^−z/1+e^−z

And since 1−σ(z))=1−1/(1+e^−z)=e^−z/(1+e^−z) this can be rewritten as:
∂y/∂z=1/1+e^−z.e^−z/1+e^−z=σ(z)*(1−σ(z))=y(1−y)
"""

# Define the logistic derivative
def logistic_derivative(z):
	return logistic(z) * (1 - logistic(z))

# Plot the derivative of the logistic function
plt.clf()
z = np.linspace(-6,6,100)
plt.plot(z, logistic_derivative(z), 'r-')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$\\frac{\\partial \\sigma(z)}{\\partial z}$', fontsize=15)
plt.title('derivative of the logistic function')
plt.grid()
plt.show()

"""
Logistic regression (classification)

While the previous tutorial described a very simple one-input-one-output linear regression model, this tutorial will describe a 2-class classification neural network with two input dimensions. This model is known in statistics as the logistic regression model. This network can be represented graphically as:

X_1 --- w_1 ---
               |
               |--------> Y
               |
X_2 --- w_2 ---

"""

"""
Define the class distributions.

In this example the target classes t will be generated from 2 class distributions: blue (t=1) and red (t=0).
Samples from both classes are sampled from their respective distributions. These samples are plotted in the figure below. Note that X is a N×2 matrix of individual input samples x_i, and that t is a corresponding N×1 vector of target values t_i.
"""

# Define and generate the samples
# Reset the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)
nb_of_samples_per_class = 20  # The number of sample in each class
red_mean = [-1,0]  # The mean of the red class
blue_mean = [1,0]  # The mean of the blue class
std_dev = 1.2  # standard deviation of both classes
# Generate samples from both classes
x_red = np.random.randn(nb_of_samples_per_class, 2) * std_dev + red_mean
x_blue = np.random.randn(nb_of_samples_per_class, 2) * std_dev + blue_mean

# Merge samples in set of input variables x, and corresponding set of output variables t
X = np.vstack((x_red, x_blue))
t = np.vstack((np.zeros((nb_of_samples_per_class,1)), np.ones((nb_of_samples_per_class,1))))

# Plot both classes on the x1, x2 plane
plt.clf()
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='class red')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='class blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.axis([-4, 4, -4, 4])
plt.title('red vs. blue classes in the input space')
plt.show()

"""
Logistic function and cross-entropy cost function.

Logistic function
The goal is to predict the target class t from the input values x.
The network is defined as having an input x=[x_1,x_2] which gets transformed by the weights w=[w_1,w_2] to generate the probability that sample x belongs to class t=1.
This probability P(t=1|x,w) is represented by the output y of the network computed as y=σ(x*w^T).
σ is the logistic function and is defined as:
σ(z)=1/1+e^−z

This logistic function and its derivative are explained in detail above.
The logistic function is implemented below by the logistic(z) method.

Cross-entropy cost function
The cost function used to optimize the classification is the cross-entropy error function . And is defined for sample i as:
ξ(t_i,y_i)=−t_i.log(y_i)−(1−t_i).log(1−y_i)

Which will give ξ(t,y)=−∑_{i=1,n}[t_i.log(y_i)+(1−t_i).log(1−y_i)]
if we sum over all N samples.

The explanation and derivative of this cost function are given above. The cost function is implemented below by the cost(y, t) method, and its output with respect to the parameters w over all samples x is plotted in the code below.

The neural network output is implemented by the nn(x, w) method, and the neural network prediction by the nn_predict(x,w) method.
"""

# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
# 'iol' = input, logistic output
def nniol(x, w): 
    return logistic(x.dot(w.T))

# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(x,w): 
    return np.around(nniol(x,w))
    
# Define the cross-entropy cost function
def cost_ce(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

# Define a vector of weights for which we want to plot the cost
plt.clf()
nb_of_ws = 100 # compute the cost nb_of_ws times in each dimension
ws1 = np.linspace(-5, 5, num=nb_of_ws) # weight 1
ws2 = np.linspace(-5, 5, num=nb_of_ws) # weight 2
ws_x, ws_y = np.meshgrid(ws1, ws2) # generate grid
cost_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize cost matrix
# Fill the cost matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        cost_ws[i,j] = cost_ce(nniol(X, np.asmatrix([ws_x[i,j], ws_y[i,j]])) , t)

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

The gradient descent algorithm works by taking the derivative of the cost function ξ with respect to the parameters, and updates the parameters in the direction of the negative gradient.

The parameters w are updated by taking steps proportional to the negative of the gradient: w(k+1)=w(k)−Δw(k+1).
Δw is defined as: Δw=μ.∂ξ/∂w with μ the learning rate.

∂ξ_i/∂w, for each sample i is computed as follows:

∂ξ_i/∂w=∂z_i/∂w.∂y_i/∂z_i.∂ξ_i/∂y_i
Where y_i=σ(z_i) is the output of the logistic neuron,
and z_i=x_i*w^T the input to the logistic neuron.

∂ξ_i/∂y_i can be calculated as:
∂ξ_i/∂y_i=y_i−t_i/y_i(1−y_i)

∂yi/∂zi can be calculated as:
∂y_i/∂z_i=y_i(1−y_i)

zi/∂w can be calculated as:
∂z/∂w=∂(x*w)∂w=x

Bringing this together we can write:

∂ξ_i/∂w=∂z_i/∂w.∂y_i/∂z_i.∂ξ_i/∂y_i
=x*y_i(1−y_i)*y_i−t_i/(y_i(1−y_i))=x*(y_i−t_i)

Notice how this gradient is the same (negating the constant factor) as the gradient of the squared error regression.

So the full update function Δw_j for each weight will become

Δw_j=μ*∂ξ_i/∂w_j=μ*x_j*(y_i−t_i)

In the batch processing, we just add up all the gradients for each sample:

Δw_j=μ*∑_{i=1,N}x_i,j(y_i−t_i)

To start out the gradient descent algorithm, you typically start with picking the initial parameters at random and start updating these parameters according to the delta rule with Δw until convergence.

The gradient ∂ξ/∂w is implemented by the gradient(w, x, t) function. Δw is computed by the delta_w(w_k, x, t, learning_rate).
"""

# define the gradient function.
def gradient_ce(w, x, t): 
    return (nniol(x, w) - t).T * x

# define the update function delta w which returns the 
#  delta w for each weight in a vector
def delta_w_ce(w_k, x, t, learning_rate):
    return learning_rate * gradient_ce(w_k, x, t)
# ?? why no .sum here
"""
Gradient descent updates.
#
Gradient descent is run on the example inputs X and targets t for 10 iterations. The first 3 iterations are shown in the plotted figure. The blue dots represent the weight parameter values w(k) at iteration k.
"""

# Set the initial weight parameter
w = np.asmatrix([-4, -2])
# Set the learning rate
learning_rate = 0.05

# Start the gradient descent updates and plot the iterations
nb_of_iterations = 10  # Number of gradient descent updates
w_iter = [w]  # List to store the weight values over the iterations
for i in range(nb_of_iterations):
    dw = delta_w_ce(w, X, t, learning_rate)  # Get the delta w update
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
    # Plot the weight-cost value and the line that represents the update
    plt.plot(w1[0,0], w1[0,1], 'bo')  # Plot the weight cost value
    plt.plot([w1[0,0], w2[0,0]], [w1[0,1], w2[0,1]], 'b-')
    plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'.format(i), color='b')
w1 = w_iter[3]  
# Plot the last weight
plt.plot(w1[0,0], w1[0,1], 'bo')
plt.text(w1[0,0]-0.2, w1[0,1]+0.4, '$w({})$'.format(4), color='b') 
# Show figure
plt.xlabel('$w_1$', fontsize=15)
plt.ylabel('$w_2$', fontsize=15)
plt.title('Gradient descent updates on cost surface')
plt.grid()
plt.show()

"""
Visualization of the trained classifier.

The resulting decision boundary of running gradient descent on the example inputs X and targets t is shown in the next plot.
The background color refers to the classification decision of the trained classifier. Note that since this decision plane is linear that not all examples can be classified correctly. Two blue dots will be misclassified as red, and four red spots will be misclassified as blue.

Note that the decision boundary goes through the point (0,0) since we don't have a bias parameter on the logistic output unit.
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
        classification_plane[i,j] = nn_predict(np.asmatrix([xx[i,j], yy[i,j]]) , w)
# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30)])

# Plot the classification plane with decision boundary and input samples
plt.contourf(xx, yy, classification_plane, cmap=cmap)
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='target red')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='target blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.title('red vs. blue classification boundary')
plt.show()



# Define the softmax function
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
fig = plt.figure()
# Plot the cost function surface for t=1
"""
ax = fig.gca(projection='3d')
surf = ax.plot_surface(zs_1, zs_2, y[:,:,0], linewidth=0, cmap=cm.coolwarm)
ax.view_init(elev=30, azim=70)
cbar = fig.colorbar(surf)
ax.set_xlabel('$z_1$', fontsize=15)
ax.set_ylabel('$z_2$', fontsize=15)
ax.set_zlabel('$y_1$', fontsize=15)
ax.set_title ('$P(t=1|\mathbf{z})$')
cbar.ax.set_ylabel('$P(t=1|\mathbf{z})$', fontsize=15)
plt.grid()
#plt.show()
"""
