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

    model = build_model_rand(500)

    history = model.direct_training((X, y), lambda_=0, p_d=0, p_dc=0.1)

    print(history["loss_tr"])
    print(model.beta_b)

    print(model_loss(model, MEE(), X, y))
    print(model_loss(model, MEE(), test_x, test_y))