import numpy as np

class Regularizer():

    def __init__(self):
        pass

    def regularize(self, weights):
        raise NotImplementedError()


class L2_regularizer(Regularizer):

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def regularize(self, weights):
        return -self.lambda_*2*weights

class L1_regularization(Regularizer):

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def regularize(self, weights):
        return -self.lambda_*np.sign(weights)