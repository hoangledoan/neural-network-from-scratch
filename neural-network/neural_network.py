import numpy as np
# import nnfs
# from nnfs.datasets import spiral_data


class Dense():

    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs_tensor) -> None:
        self.output = np.dot(inputs_tensor, self.weights) + self.biases


class ActivationFunction():

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


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # One-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)
print(average_loss)
