from abc import ABC

from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Abstract class of optimizers
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def clear(self):
        """
        Clear method necessary to clean some attributes that have to be reinitialized
        between a training phase and another
        """
        raise NotImplementedError()

    @abstractmethod
    def optimize(self):
        """
        Method that given deltas optimized it wrt regularizations and other optimizations techniques

        Returns:
            deltas: (list) list of tuples (delta weights, delta biases) of optimized deltas
        """
        raise NotImplementedError()


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, alpha, decay=None):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.decay = decay

        self.memory = None # old deltas

    def clear(self):
        self.memory = None

    def optimize(self, deltas, regs, epoch):

        # decaying learning rate
        if self.decay:
            tau, learning_rate_tau = self.decay
            if epoch <= tau:
                a = epoch/tau
                l_r = (1 - a)*self.learning_rate + a*learning_rate_tau
            else:
                l_r = learning_rate_tau
        else:
            l_r = self.learning_rate

        if self.memory:
            for i, (delta_w, delta_b) in enumerate(deltas):
                old_delta_w, old_delta_b = self.memory[i]
                reg_w, reg_b = regs[i]
                deltas[i] = (l_r*delta_w + self.alpha*old_delta_w + reg_w, l_r*delta_b + self.alpha*old_delta_b + reg_b)
        else:
            for i, (delta_w, delta_b) in enumerate(deltas):
                reg_w, reg_b = regs[i]
                deltas[i] = (l_r*delta_w + reg_w, l_r*delta_b + reg_b)

        self.memory = deltas

        return deltas


optimizer_functions = {
    "sgd" : StochasticGradientDescent(0.1, 0.3, None)
}