import numpy as np

from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    """
    Abstract class of the activation functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute(self, x:np.ndarray):
        """
        Compute the activation function
        
        Args:
            x: (np.ndarray) input of the activation function

        Returns:
            returns: (np.ndarray) output of the activation function
        """

        pass

    @abstractmethod
    def derivative(self, x:np.ndarray):
        """
            Compute the derivative of the activation function
            
            Args:
                x: (np.ndarray) input of the activation function

            Returns:
                returns: (np.ndarray) derivative of the activation function
        """

        pass


class Linear(ActivationFunction):
    """ Linear activation function """

    def compute(self, x: np.ndarray):
        return x

    def derivative(self, _: np.ndarray):
        return 1


class ReLU(ActivationFunction):
    """ ReLU activation function """

    def compute(self, x: np.ndarray):
        return np.maximum(x, 0)

    def derivative(self, x: np.ndarray):
        mf = lambda y: 1 if y > 0 else 0
        mf_v = np.vectorize(mf)

        return mf_v(x)


class Sigmoid(ActivationFunction):
    """ Sigmoid activation function """

    def compute(self, x: np.ndarray):
        ones = np.ones(x.shape)
        return np.divide(ones, np.add(ones, np.exp(-x)))
    
    def derivative(self, x: np.ndarray):
        return np.multiply(self.compute(x), np.subtract(np.ones(x.shape), self.compute(x)))


class Tanh(ActivationFunction):
    """ Hyperbolic tangent function (TanH) """

    def compute(self, x: np.ndarray):
        return np.tanh(x)

    def derivative(self, x: np.ndarray):
        return 1 - np.tanh(x)**2


activation_functions = {
    'linear': Linear(),
    'relu': ReLU(),
    'sigmoid': Sigmoid(),
    'tanh': Tanh()
}