import numpy as np

def linear(input: np.ndarray):
    """ linear activation function """
    return input


def relu(input: np.ndarray):
    """ ReLU activation function """
    return np.maximum(input, 0)


def sigmoid(input: np.ndarray):
    """ Sigmoid activation function """
    ones = np.ones(input.shape)
    return np.divide(ones, np.add(ones, np.exp(-input)))


def derivate_sigmoid(input: np.ndarray):
    """ Derivative of sigmoid activation function """
    return np.multiply(sigmoid(input), np.subtract(np.ones(input.shape), sigmoid(input)))


def tanh(input: np.ndarray):
    """ Hyperbolic tangent function (TanH) """
    return np.tanh(input)


def derivative_relu(input: np.ndarray):
    """ Derivative of ReLU activation function """
    mf = lambda x: 1 if x > 0 else 0
    mf_v = np.vectorize(mf)

    return mf_v(input)


activation_functions = {
    'linear': linear,
    'relu': relu,
    'sigmoid': sigmoid,
    #'tahn': tahn
}


derivate_activation_functions = {
    'linear': lambda _: 1,
    'relu': derivative_relu,
    'sigmoid': derivate_sigmoid,
    #'tahn': derivate_tahn
}