from numpy import histogram
from feed_forward import *
from optimizers import StochasticGradientDescent
from datasets import *
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

best_params = grid_search_cv(monk_2_build_model, (X, y), {"learning_rate": [0.6, 0.7], "alpha": [0.8, 0.9], "epochs":[5, 10], "batch_size": "full"})

print(best_params)

train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.33, random_state=42)

best_comp_params, best_train_params = split_train_params(best_params)

model=monk_2_build_model(**best_comp_params)

history = model.training((train_x, train_y), (valid_x, valid_y), **best_train_params, batch_size="full")

plt.plot(history["loss_tr"])
plt.plot(history["loss_vl"])
plt.show()