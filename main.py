from numpy import histogram
from mlprj.ensamble import *
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

    nn = RandomizedNetwork(10, RandomizedLayer(hidden_neurons, "sigmoid"), 2)
    nn.compile(loss=MSE())

    return nn


if __name__ == '__main__':

    X, test_x, y, test_y, preprocesser = read_cup()

    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.1, random_state=42)

    model = Network(10, [Layer(20, "relu", "gaussian"), Layer(10, "sigmoid", "gaussian"), Layer(2, "linear", "gaussian")])

    model.compile(loss=MSE(), regularizer=L2Regularizer(0.000025), optimizer=StochasticGradientDescent(0.2, 0.175))

    model.training((train_x, train_y), (valid_x, valid_y), batch_size=32, early_stopping=20, verbose=True)

    print(model_loss(model, MSE(), test_x, test_y))
    print(model_loss(model, MEE(), test_x, test_y))
