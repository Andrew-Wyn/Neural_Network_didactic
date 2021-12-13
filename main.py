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

    model1 = build_model(0.3, 0.5)

    model1.training((train_x, train_y), (test_x, test_y), 100, "full", verbose=True)


    model2 = build_model(0.4, 0.1)

    model2.training((train_x, train_y), (test_x, test_y), 100, "full", verbose=True)

    ens = Ensamble([model1, model2])

    print(model_loss(ens, MSE(), test_x, test_y))