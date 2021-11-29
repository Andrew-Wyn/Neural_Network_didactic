import numpy as np

class GaussianInitializer:
    def __init__(self):
        pass

    def initialize(self, input_dim, output_dim):
        """Weights Gaussian initialization"""
        return np.random.normal(size = (output_dim, input_dim)), np.random.normal(size = output_dim)


class UniformInitializer:
    def __init__(self, min_value=-0.5, max_value=0.5):
        self.min_value = min_value
        self.max_value = max_value

    def initialize(self, input_dim, output_dim):
        """Weights Uniform initialization"""
        return np.random.uniform(low = self.min_value, high = self.max_value, size = (output_dim, input_dim)), np.random.uniform(low = self.min_value, high = self.max_value, size = output_dim)


class ConstantInitializer:
    def __init__(self, value=0.1):
        self.value = value
    def constant_initialization(self, input_dim, output_dim):
        """Weights constant initialization"""
        return np.full(shape = (output_dim, input_dim), fill_value=self.value), np.full(shape = output_dim, fill_value=self.value)


initialization_functions = {
    'gaussian': GaussianInitializer(),
    'uniform': UniformInitializer(),
    'constant_initialization' : ConstantInitializer(),
}
