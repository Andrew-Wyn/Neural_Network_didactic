from numpy import histogram
from mlprj.feed_forward import *
from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.randomized_nn import *
from mlprj.regularizers import *
from mlprj.initializers import *

from mlprj.utility import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def build_model_rand(hidden_neurons):

    nn = RandomizedNetwork(17,
    [
    RandomizedLayer(hidden_neurons, "relu"),
    RandomizedLayer(1, "linear")])

    nn.compile(loss=MSE())

    return nn


def build_model(learning_rate, alpha):

    nn = Network(17,
    [
    Layer(5, "relu", GaussianInitializer()),
    Layer(1, "sigmoid", GaussianInitializer())]
    )

    nn.compile(loss=MSE(), regularizer=L2Regularizer(0), optimizer=StochasticGradientDescent(learning_rate, alpha))

    return nn


if __name__ == '__main__':

    train_x, test_x, train_y, test_y = read_monk(1)

    best_params = grid_search_cv(build_model, (train_x, train_y), {
        "learning_rate": [0.3, 0.4],
        "alpha": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "epochs": 10,
        "batch_size": "full"
        }, 
    k_folds=5)
    
    print(best_params)

    best_params_others, best_params_training = split_train_params(best_params)

    print(best_params_others)