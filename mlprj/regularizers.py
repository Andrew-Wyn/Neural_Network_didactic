from abc import ABC
import numpy as np

from abc import ABC, abstractmethod

class Regularizer(ABC):
    """
    Abstract regularizer functions
    """

    def __init__(self):
        pass

    @abstractmethod
    def regularize(self, weights):
        """
        Regularize weight and return the value associated to the deltas wrt the weights itself

        Args:
            weights: (np.ndarray) weights
        Returns:
            ret: (np.ndarray) deltas of the regularized weights
        """
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