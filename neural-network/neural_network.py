from cmath import exp
from xml.dom import xmlbuilder
import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons)

def first_layer(input_data, weight1, biases):
    return np.dot(input_data, np.array(weight1).T) + biases


def second_layer(output_first_layer, weight2, biases):
    return np.dot(output_first_layer, np.array(weight2).T) + biases


def sigmoid_activation(x):
    """Return a value from 0 to 1."""
    return 1/(1+exp(x))

def relu_activation(x):
    """Return x if x > 0, else 0."""
    return max(0, x)
