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

    train_x, test_x, train_y, test_y = read_monk(2)

    model = build_model_rand(1500)

    model.direct_training((train_x, train_y), validation=(test_x, test_y), lambda_=0, p_d=1, p_dc=0.8, verbose=True)

    print(model_accuracy(model, train_x, train_y))
    print(model_accuracy(model, test_x, test_y))

    """

    monk_1_params = {
        "learning_rate": [0.7, 0.8],
        "alpha": [0.4, 0.4],
        "epochs": 10,
        "batch_size": "full"
    }

    monk_1_best_params = grid_search(build_model, (train_x, train_y), (test_x, test_y), monk_1_params, path="monk1.csv")
    monk_1_best_params_other, monk_1_best_params_training = split_train_params(monk_1_best_params)
    print(monk_1_best_params_other, monk_1_best_params_training)
    """