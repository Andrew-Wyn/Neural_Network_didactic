from numpy import histogram
from mlprj.feed_forward import *
from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *

from mlprj.utility import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train_x, test_x, train_y, test_y = read_monk(2)

    def build_model(lambda_, alpha):
        nn = Network(
            17,
            [Layer(4, "relu", "uniform", {"distribution_range":(-0.25, 0.25)}),
            Layer(1,"sigmoid", "uniform", {"distribution_range":(-0.25, 0.25)})]
        )

        nn.compile(loss=MSE(), regularizer=L2_regularizer(0), optimizer=StochasticGradientDescent(0.8, 0.8))

        return nn

    best_params = grid_search(build_model, (train_x, train_y), (test_x, test_y), {"lambda_":[0.1, 0.2], "alpha":[0.1, 0.2], "epochs":200, "batch_size":64})