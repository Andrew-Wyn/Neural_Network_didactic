class Optimizer:
    def __init__(self):
        pass

    def optimize(self):
        raise NotImplementedError()

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, alpha):
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.memory = None # old deltas

    def reset(self):
        self.memory = None

    def optimize(self, deltas, regs):        
        if self.memory:
            for i, (delta_w, delta_b) in enumerate(deltas):
                old_delta_w, old_delta_b = self.memory[i]
                reg_w, reg_b = regs[i]
                deltas[i] = (self.learning_rate*delta_w + self.alpha*old_delta_w + reg_w, self.learning_rate*delta_b + self.alpha*old_delta_b + reg_b)
        else:
            for i, (delta_w, delta_b) in enumerate(deltas):
                reg_w, reg_b = regs[i]
                deltas[i] = (self.learning_rate*delta_w + reg_w, self.learning_rate*delta_b)

        self.memory = deltas

        return deltas