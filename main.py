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
    X, y = read_cup()
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.30)

    #train_x, test_x, train_y, test_y = read_monk(3)

    nn = RandomizedNetwork(10,
    [
    RandomizedLayer(3000, "relu"),
    RandomizedLayer(2, "linear")])

    nn.compile(loss=MSE(), regularizer=L2Regularizer(0), optimizer=StochasticGradientDescent(0.9, 0.9))

    nn.direct_training((train_x, train_y), (valid_x, valid_y), lambda_=150, verbose=True)

    #print(model_accuracy(nn, train_x, train_y))
    #print(model_accuracy(nn, valid_x, valid_y))

    # best_params = grid_search_cv(build_model, (train_x, train_y), {"lambda_":[0.5, 0.7], "alpha":[0.5, 0.7], "epochs":[50, 10], "batch_size":"full"})

    #print(best_params)