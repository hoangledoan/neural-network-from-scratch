import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Dense:

    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None
        self.input = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs_tensor) -> None:
        self.input = inputs_tensor
        self.output = np.dot(inputs_tensor, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0)


class ActivationFunction:

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


class Loss1:

    def average_loss(self, loss):
        return np.mean(loss)

    def categorical_loss(self, true_value, predicted_value):

        # Prevent divide by 0
        predicted_value = np.clip(predicted_value, 1e-7, 1 - 1e-7)

        # If the true labels are not encoded
        if len(true_value.shape) == 1:
            correct_confidences = predicted_value[range(len(predicted_value)),
                                                  true_value]
        # If the true labels are encoded
        if len(true_value.shape) == 2:
            correct_confidences = np.sum(true_value * predicted_value, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return self.average_loss(negative_log_likelihoods)


if __name__ == "__main__":
    X, y = spiral_data(samples=2, classes=4)
    layer1 = Dense(2, 5)
    layer1.forward(X)
    layer2 = Dense(5, 10)
    layer2.forward(layer1.output)
    activation = ActivationFunction()
    activation.softmax_activation(layer2.output)
    layer2.backward(layer2.output)
    # loss = Loss1()
    # cate_loss = loss.categorical_loss(y, activation.output)
    # print(cate_loss)
    print(layer2.dw)
