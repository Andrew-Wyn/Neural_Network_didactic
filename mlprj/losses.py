from abc import ABC
import numpy as np

from abc import ABC, abstractmethod

class Loss(ABC):
    """
    Abstract class implementing Loss functions
    """

    def __init__(self):
        pass
    
    @abstractmethod
    def compute(self, target, output):
        """
        Compute the loss function

        Args:
            target: (np.ndarray) target values
            output: (np.ndarray) output values returned from a model

        Returns:
            ret: (np.ndarray) the loss
        """
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, target, output):
        """
        Compute the derivative of the loss function

        Args:
            target: (np.ndarray) target values
            output: (np.ndarray) output values returned from a model
        Returns:
            ret: (np.ndarray) the derivative of the loss
        """
        raise NotImplementedError()


class MSE(Loss):

    def __init__(self):
        pass

    def compute(self, target, output):
        diff = target - output
        return np.linalg.norm(diff)**2

    def derivative(self, target, output):
        return -2*(target - output)


class MEE(Loss):
    
    def __init__(self):
        pass

    def compute(self, target, output):
        diff = target - output
        return np.linalg.norm(diff)

    def derivative(self, target, output):
        diff = target - output
        return -(diff)/(np.sqrt((diff)**2))


class BinaryCrossEntropy(Loss):

    def __init__(self):
        pass

    def compute(self, target, output):
        return -np.sum((target*np.log(output) + (1-target)*np.log(1-output)))

    def derivative(self, target, output):
        return -((target-output)/(output*(1-output)))


loss_functions = {
    'mse': MSE(),
    'mee': MEE(),
    'bce' : BinaryCrossEntropy(),
}