from abc import ABC
import numpy as np

from abc import ABC, abstractmethod

class Loss(ABC):

    def __init__(self):
        pass
    
    @abstractmethod
    def compute(self, target, output):
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, target, output):
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
        return -(target*np.log(output) + (1-target)*np.log(1-output))

    def derivative(self, target, output):
        return -((target-output)/(output*(1-output)))


loss_functions = {
    'mse': MSE(),
    'mee': MEE(),
    'bce' : BinaryCrossEntropy(),
}