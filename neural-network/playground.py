import numpy as np

# softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4],
#                             [0.02, 0.9, 0.08]])
# class_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
# print(np.sum(softmax_outputs * class_targets, axis=1))
a = np.array([[1, 2, 3], [3, 4, 5]])
b = np.array([[0, 1, 0], [1, 0, 0]])
print(sum(a * b))
