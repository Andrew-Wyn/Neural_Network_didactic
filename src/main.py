from feed_forward import *
from utility import *

import matplotlib.pyplot as plt

train_x, train_y = read_monk(2)

print(train_x.shape)
print(train_y.shape)

"""
nn = Network(10, [
    Layer(128, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(64, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(32, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(16, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(1, "linear", 'uniform', {"distribution_range": (-0.25, 0.25)}),
])

nn.compile()
"""

nn = Network(17, [
    Layer(2, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(1, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)})
])

nn.compile()

history = nn.training(train_x, train_y, 500, 0.75, 0.8, 64)
plt.plot(history["loss"])
plt.show()