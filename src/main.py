from feed_forward import *
from read_monk import *

import matplotlib.pyplot as plt

train_x, train_y = read_monk(2)

print(train_x.shape)
print(train_y.shape)

nn = Network(17, [
    Layer(2, "relu", 'uniform', {"distribution_range": (-0.25, 0.25)}),
    Layer(1, "sigmoid", 'uniform', {"distribution_range": (-0.25, 0.25)}),
])

nn.compile()

history = nn.training(train_x, train_y, 500, 0.75, 0.8, len(train_x))
plt.plot(history["loss"])
plt.show()
