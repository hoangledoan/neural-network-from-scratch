import numpy as np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T
# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

layer_output = np.dot(inputs, weights) + biases
relu_output = np.maximum(layer_output, 0)

drelu = relu_output.copy()
drelu[layer_output <= 0] = 0

dinput = np.dot(drelu, weights.T)
dweight = np.dot(inputs.T, drelu)

dbiases = np.sum(drelu, axis=0)
print(drelu)
print(dbiases)