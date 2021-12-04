import numpy as np

class Loss:

    def __init__(self):
        pass

    def compute(self, target, output):
        raise NotImplementedError()

    def derivate(self, target, output):
        raise NotImplementedError()


class MSE(Loss):

    def __init__(self):
        pass

    def compute(self, target, output):
        diff = target - output
        return np.linalg.norm(diff)**2

    def derivate(self, target, output):
        return -2*(target - output)


class MEE(Loss):
    
    def __init__(self):
        pass

    def compute(self, target, output):
        diff = target - output
        return np.linalg.norm(diff)

    def derivate(self, target, output):
        diff = target - output
        return -(diff)/(np.sqrt((diff)**2))