from feed_forward import *
from utility import *
from model_selection import *
from losses import *
from regularizers import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X, y = read_monk(2)

print(X.shape)
print(y.shape)

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
    Layer(2, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(1, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)})
])

train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.33, random_state=42)

best_param = grid_search(nn, (train_x, train_y), (valid_x, valid_y), loss=MSE(), regularizer=L2_regularizer(0), fixed_params={"epochs": 300, "batch_size": 64}, search_params={"learning_rate": [0.1, 0.3, 0.6, 0.9], "alpha": [0.1, 0.3, 0.6, 0.9]})
print(best_param)

best_lr = best_param["learning_rate"]
best_a = best_param["alpha"]

best_param = grid_search(nn, (train_x, train_y), (valid_x, valid_y), loss=MSE(), regularizer=L2_regularizer(0), fixed_params={"epochs": 300, "batch_size": 64}, search_params={"learning_rate": [best_lr-0.1, best_lr-0.005, best_lr+0.005, best_lr+0.1 ], "alpha": [best_a-0.1, best_a-0.005, best_a+0.005, best_a+0.1 ]})
print(best_param)


nn.compile(loss=MSE(), regularizer=L2_regularizer(0))
history = nn.training((train_x, train_y), (valid_x, valid_y), **{"epochs": 300, "batch_size": 64}, **best_param)
plt.plot(history["loss_tr"])
plt.show()