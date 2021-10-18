import numpy as np
from activation_functions import activation_functions

class Layer:
    def __init__(self, weights_matrix: np.ndarray, bias: np.ndarray, activation: str):
        """

        """
        self.weights_matrix = weights_matrix
        self.bias = bias
        self.activation = activation_functions[activation]

    def forward_step (self, input: np.ndarray):
        net = np.matmul(self.weights_matrix, self.input)
        net = np.add(self.net, self.bias)
        out = self.activation(net)
        return out
