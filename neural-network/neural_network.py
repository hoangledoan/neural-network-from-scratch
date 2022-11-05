import numpy as np
import nnfs  # pylint: disable=import-error
from nnfs.datasets import spiral_data  # pylint: disable=import-error

nnfs.init()


class Dense:
    """A neuron network layer."""

    def __init__(self, n_inputs, n_neurons) -> None:
        """Create a neuron network layer.

        Args:
            n_inputs: the numpy array input.
            n_neurons: the number of expecting output.
        """
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None
        self.input = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs_tensor) -> None:
        """Calculate the output.

        Args:
            inputs_tensor: the numpy array input.
        """
        self.input = inputs_tensor
        self.output = np.dot(inputs_tensor, self.weights) + self.biases

    def backward(self, dvalues) -> None:
        """Calculate the derivative of the input in respect to the weights and the biases.

        Args:
            dvalues: the derivatived tensor.
        """
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0)


class ActivationRelu:
    """The Relu activation function."""

    def __init__(self) -> None:
        self.output = None
        self.input = None
        self.drelu = None

    def relu_activation(self, input_tensor) -> None:
        """Return x if x > 0, else 0."""
        self.input = input_tensor
        self.output = np.maximum(0, input_tensor)

    def backward(self, dvalues) -> None:
        """Calculate the derivative in respect to the relu activation function.

        Args:
            dvalues: numpy array input.
        """
        self.drelu = dvalues.copy()

        # drelu = x for x > 0, else = 0
        self.drelu[self.input <= 0] = 0


class ActivationSoftmax:
    """The softmax activation function."""

    def __init__(self) -> None:
        self.output = None
        self.dinputs = None

    def softmax_activation(self, input_tensor):
        """Return a list of number which add up to 1."""
        sum_list = [np.exp(number) for number in input_tensor]
        e_sum = sum(sum_list)
        self.output = [np.exp(number) / e_sum for number in input_tensor]

    def backward(self, dvalues):
        """Calculate the derivative of the softmax activation function."""

        # Each input to the softmax impacts each of the outputs, so we need to calculate the partial derivative of each output with respect to each input (Jacobian matrix)
        # dS_ij/dz_ik = S_ij * (impulsantwort_jk - S_ik) = S_ij * impulsantwort_jk - S_ij * S_ik
        # impuls_antwort_jk = 1 when j = k, else 0

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate ouputs and gradients
        for index, (single_output,
                    single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix of the output (S_ij * S_ik)
            # Iterating over the j and k indices
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T)
            # S_ij * impulsantwort_jk - S_ij * S_ik
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class ActivationSigmoid:
    """The sigmoid activation function."""

    def __init__(self) -> None:
        self.output = None

    def sigmoid_activation(self, input_tensor):
        """Return a value from 0 to 1."""
        self.output = 1 / (1 + np.exp(input_tensor))


class Loss1:
    """General loss function."""

    def average_loss(self, loss_tensor):
        """Calculate the average loss in a tensor.

        Args:
            loss_tensor: the input loss tensor.
        """
        return np.mean(loss_tensor)


class CategoricalLoss(Loss1):
    """The categorical loss."""

    def __init__(self) -> None:
        self.dinputs = None

    def categorical_loss(self, true_value, predicted_value):
        """Calculate the categorical loss.

        Args:
            true_value: the true labels tensor.
            predicted_value: the predict labels tensor.
        """
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

    def backward(self, dvalues, true_value):
        """Calculate the derivative of the categorical loss."""
        # dE/dy_hat = -dy/dy_hat
        if true_value.shape == 1:
            true_value = np.eye(dvalues[0])[true_value]
        self.dinputs = -true_value / dvalues

        # Normalize the dinputs for the later sum
        self.dinputs = self.dinputs / len(dvalues)


if __name__ == "__main__":
    X, y = spiral_data(samples=2, classes=4)
    layer1 = Dense(2, 5)
    layer1.forward(X)
    layer2 = Dense(5, 10)
    layer2.forward(layer1.output)
    activation = ActivationFunction()
    activation.softmax_activation(layer2.output)
    layer2.backward(layer2.output)
    loss = Loss1()
    # cate_loss = loss.categorical_loss(y, activation.output)
    # print(cate_loss)
    print(layer2.dw)
