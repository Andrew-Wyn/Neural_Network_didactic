import numpy as np

class Layer:
    def __init__(self, weights_matrix: np.ndarray, bias: np.ndarray):
        """

        """
        self.weights_matrix = weights_matrix
        self.bias = bias
        
