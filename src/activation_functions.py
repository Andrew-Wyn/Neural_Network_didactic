import numpy as np

def linear(input: np.ndarray):
    """ linear activation function """
    return input

def relu(input: np.ndarray):
    """ ReLU activation function """
    return np.maximum(input, 0)

def sigmoid(input: np.ndarray):
    """ Sigmoid activation function """
    ones = [1.] * len(input)
    return np.divide(ones, np.add(ones, np.exp(-input)))

activation_functions = { 'linear': linear, 'relu': relu, 'sigmoid':sigmoid}
