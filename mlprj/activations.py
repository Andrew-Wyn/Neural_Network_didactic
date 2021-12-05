import numpy as np


def linear(x: np.ndarray):
    """ linear activation function """
    return x


def relu(x: np.ndarray):
    """ ReLU activation function """
    return np.maximum(x, 0)


def sigmoid(x: np.ndarray):
    """ Sigmoid activation function """
    ones = np.ones(x.shape)
    return np.divide(ones, np.add(ones, np.exp(-x)))


def derivate_sigmoid(x: np.ndarray):
    """ Derivative of sigmoid activation function """
    return np.multiply(sigmoid(x), np.subtract(np.ones(x.shape), sigmoid(x)))


def tanh(x: np.ndarray):
    """ Hyperbolic tangent function (TanH) """
    return np.tanh(x)


def derivative_tanh(x: np.ndarray):
  """ Derivative of hyperbolic tangent function (TanH) """
  return 1 - np.tanh(x)**2


def derivative_relu(x: np.ndarray):
    """ Derivative of ReLU activation function """
    mf = lambda y: 1 if y > 0 else 0
    mf_v = np.vectorize(mf)

    return mf_v(x)


activation_functions = {
    'linear': linear,
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh
}


derivate_activation_functions = {
    'linear': lambda _: 1,
    'relu': derivative_relu,
    'sigmoid': derivate_sigmoid,
    'tanh': derivative_tanh
}