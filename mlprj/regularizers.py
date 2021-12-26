from abc import ABC
import numpy as np

from abc import ABC, abstractmethod

class Regularizer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def regularize(self, weights):
        raise NotImplementedError()


class L2Regularizer(Regularizer):

    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_

    def regularize(self, weights):
        return -self.lambda_*2*weights


class L1Regularizer(Regularizer):

    def __init__(self, lambda_=0.1):
        self.lambda_ = lambda_

    def regularize(self, weights):
        return -self.lambda_*np.sign(weights)


regularizer_functions = {
    "l1" : L1Regularizer(),
    "l2" : L2Regularizer()
}