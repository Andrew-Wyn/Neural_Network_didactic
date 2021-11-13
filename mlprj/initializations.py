import numpy as np


def gaussian_initialization(input_dim, output_dim):
    """Weights Gaussian initialization"""
    return np.random.normal(size = (output_dim, input_dim)), np.random.normal(size = output_dim)


def uniform_initialization(input_dim, output_dim, distribution_range):
    """Weights Uniform initialization"""
    a, b = distribution_range
    return np.random.uniform(low = a, high = b, size = (output_dim, input_dim)), np.random.uniform(low = a, high= b, size = output_dim)


def constant_initialization(input_dim, output_dim, value):
    """Weights constant initialization"""
    return np.full(shape = (output_dim, input_dim), fill_value=value), np.full(shape = output_dim, fill_value=value)


initialization_functions = {
    'gaussian': gaussian_initialization,
    'uniform': uniform_initialization,
    'constant_initialization' : constant_initialization,
}