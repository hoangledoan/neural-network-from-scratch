import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Dense():

    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs_tensor) -> None:
        self.output = np.dot(inputs_tensor, self.weights) + self.biases


class Activation_Function():

    def __init__(self) -> None:
        self.output = None

    def sigmoid_activation(self, input_tensor):
        """Return a value from 0 to 1."""
        self.output = 1 / (1 + np.exp(input_tensor))

    def relu_activation(self, input_tensor):
        """Return x if x > 0, else 0."""
        self.output = np.maximum(0, input_tensor)

    def softmax_activation(self, input_tensor):
        """Return a list of number which add up to 1."""
        sum_list = [np.exp(number) for number in input_tensor]
        e_sum = sum(sum_list)
        self.output = [np.exp(number) / e_sum for number in input_tensor]


if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Dense(2, 3)
    dense1.forward(X)
    dense2 = Dense(3, 6)
    dense2.forward(dense1.output)
    softmax = Activation_Function()
    softmax.softmax_activation(dense2.output)
    print(softmax.output)