from feed_forward import *
from optimizers import StochasticGradientDescent
from utility import *
from model_selection import *
from losses import *
from regularizers import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def monk_2_build_model(learning_rate, alpha):
    nn = Network(17, [
        Layer(2, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
        Layer(1, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)})
    ])

    nn.compile(loss=MSE(), regularizer=L2_regularizer(0), optimizer=StochasticGradientDescent(learning_rate, alpha))

    return nn


X, y = read_monk(2)

print(X.shape)
print(y.shape)

# loss_tr, loss_vl = cross_validation(monk_2_build_model, (X, y), {"learning_rate": 0.6, "alpha": 0.8, "epochs":200, "batch_size": "full"},2)

#plt.plot(loss_tr)
#plt.plot(loss_vl)
#plt.show()

best_params = grid_search_cv(monk_2_build_model, (X, y), {"learning_rate": [0.6, 0.7], "alpha": [0.8, 0.9], "epochs":100, "batch_size": "full"})

print(best_params)
