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

def build_model(hidden_neurons):

    nn = RandomizedNetwork(10,
    [
    RandomizedLayer(hidden_neurons, "relu"),
    RandomizedLayer(2, "linear")])

    nn.compile(loss=MEE())

    return nn

if __name__ == '__main__':
    X, y = read_cup()
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.10)

    best_params = grid_search_cv(build_model, (train_x, train_y), {"hidden_neurons":[200], "lambda_":[0.2]}, k_folds=5, direct=True)
    
    print(best_params)

    best_params_others, best_params_training = split_train_params(best_params, direct=True)

    model = build_model(**best_params_others)

    loss_tr, loss_vl = model.direct_training((train_x, train_y), **best_params_training)

    print(f"MEE error : {model_loss(model, MEE(), test_x, test_y)}")

