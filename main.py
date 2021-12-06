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

def build_model(lambda_, alpha):

    nn = Network(
        17,
        [Layer(4, "relu", GaussianInitializer()),
        Layer(1,"sigmoid", GaussianInitializer())]
    )

    nn.compile(loss=MSE(), regularizer=L2Regularizer(0), optimizer=StochasticGradientDescent(lambda_, alpha))

    return nn

if __name__ == '__main__':
    X, test_x, y, test_y = read_monk(1)
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.10)

    nn = RandomizedNetwork(17,
    [
    RandomizedLayer(10, "tanh"),
    RandomizedLayer(100, "relu"),
    RandomizedLayer(100, "tanh"),
    RandomizedLayer(1000, "tanh"),
    RandomizedLayer(1, "sigmoid")])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(0), optimizer=StochasticGradientDescent(0.9, 0.9))

    nn.training((X, y), epochs=10000, batch_size="full", verbose=True)

    # best_params = grid_search_cv(build_model, (train_x, train_y), {"lambda_":[0.5, 0.7], "alpha":[0.5, 0.7], "epochs":[50, 10], "batch_size":"full"})

    #print(best_params)