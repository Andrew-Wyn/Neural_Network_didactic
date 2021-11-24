from numpy import histogram
from mlprj.feed_forward import *
from mlprj.optimizers import StochasticGradientDescent
from mlprj.datasets import *
from mlprj.model_selection import *
from mlprj.losses import *
from mlprj.regularizers import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def build_model(learning_rate, alpha):
    nn = Network(10, [
        Layer(128, "relu", "gaussian"),
        Layer(64, "sigmoid", "uniform", {"distribution_range":(-0.25, 0.25)}),
        Layer(128, "relu", "gaussian"),
        Layer(128, "relu", "gaussian"),
        Layer(64, "sigmoid", "uniform", {"distribution_range":(-0.25, 0.25)}),
        Layer(128, "relu", "gaussian"),
        Layer(64, "relu", "uniform", {"distribution_range":(-0.3, 0.3)}),
        Layer(32, "relu", "gaussian"),
        Layer(16, "relu", "uniform", {"distribution_range":(-0.1, 0.1)}),
        Layer(4, "relu", "gaussian"),
        Layer(64, "relu", "uniform", {"distribution_range":(-0.3, 0.3)}),
        Layer(32, "relu", "gaussian"),
        Layer(16, "relu", "uniform", {"distribution_range":(-0.1, 0.1)}),
        Layer(4, "relu", "gaussian"),
        Layer(2, "linear", "gaussian")
    ])

    nn.compile(loss=MEE(), regularizer=L2_regularizer(0.0001), optimizer=StochasticGradientDescent(learning_rate, alpha))

    return nn


X, y = read_cup()

print(X.shape)
print(y.shape)

train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.10, random_state=42)

model=build_model(0.6, 0.3)

history = model.training((train_x, train_y), (valid_x, valid_y), epochs=500, batch_size=1)

plt.plot(history["loss_tr"])
plt.plot(history["loss_vl"])
plt.show()

