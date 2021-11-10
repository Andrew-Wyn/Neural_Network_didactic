from feed_forward import *
from optimizers import StochasticGradientDescent
from utility import *
from model_selection import *
from losses import *
from regularizers import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X, y = read_monk(2)

print(X.shape)
print(y.shape)

# train test split
train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.10, random_state=42)

"""
nn = Network(10, [
    Layer(128, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(64, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(32, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(16, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(2, "linear", 'uniform', {"distribution_range": (-0.25, 0.25)}),
])

"""
nn = Network(17, [
    Layer(2, "relu", 'uniform', {"distribution_range": (0.01, 0.25)}),
    Layer(1, "sigmoid", 'uniform', {"distribution_range": (0.01, 0.25)})
])

"""
best_param = grid_search(nn, (train_x, train_y), (valid_x, valid_y), loss=MSE(), regularizer=L2_regularizer(0), fixed_params={"epochs": 300, "batch_size": 64}, search_params={"learning_rate": [0.1, 0.3, 0.6, 0.9], "alpha": [0.1, 0.3, 0.6, 0.9]})
print(best_param)
"""

nn.compile(loss=MSE(), regularizer=L2_regularizer(0), optimizer=StochasticGradientDescent(learning_rate=0.75, alpha=0.8))
history = nn.training((train_x, train_y), (valid_x, valid_y), epochs=500, batch_size=len(train_x))
plt.plot(history["loss_tr"])
plt.plot(history["loss_vl"])
plt.show()