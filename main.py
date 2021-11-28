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

train_x, test_x, train_y, test_y = read_monk(2)

nn = Network(
    17,
    [Layer(4, "relu", "uniform", {"distribution_range":(-0.25, 0.25)}),
    Layer(1,"sigmoid", "uniform", {"distribution_range":(-0.25, 0.25)})]
)

nn.compile(loss=MSE(), regularizer=L2_regularizer(0), optimizer=StochasticGradientDescent(0.8, 0.8))

history = nn.training((train_x, train_y), epochs=500, batch_size="full")

accs = model_accuracy(nn, test_x, test_y)

print(accs)

plt.plot(history["loss_tr"])
plt.show()