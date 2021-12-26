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

    nn = RandomizedNetwork(17, RandomizedLayer(hidden_neurons, "sigmoid"), 1)
    nn.compile(loss=MSE())

    return nn


if __name__ == '__main__':

    train_x, test_x, train_y, test_y = read_monk(3)

    print(cross_validation(build_model_rand, (train_x, train_y), {"hidden_neurons": 700, "lambda_": 0.5, "p_d":1, "p_dc":1}, k_folds=5, direct=True))

    model = build_model_rand(700)

    model.direct_training((train_x, train_y), validation=(test_x, test_y), lambda_=0.5, p_d=1, p_dc=1, verbose=True)

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