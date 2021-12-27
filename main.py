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

    nn = RandomizedNetwork(17, RandomizedLayer(hidden_neurons, "sigmoid"), 1)
    nn.compile(loss=MSE())

    return nn


def build_model():
    nn = Network(17, [Layer(4, "relu", UniformInitializer(-0.25, 0.25)), Layer(1, "sigmoid", UniformInitializer(-0.25, 0.25))])
    nn.compile(loss=MSE(), regularizer=L2Regularizer(0), optimizer=StochasticGradientDescent(0.7, 0.7))
    return nn

if __name__ == '__main__':

    X, test_x, y, test_y, preprocesser = read_monk(1)

    a = np.zeros(500)
    b = np.zeros(500)
    
    for i in range(10):
        model = build_model()

        history = model.training((X, y), (test_x, test_y), 500, "full", accuracy_curve=True)
        
        a += np.array(history["accuracy_tr"])
        b += np.array(history["accuracy_vl"])
        
        print(history["accuracy_tr"][-1])
        print(history["accuracy_vl"][-1])
        print()

    plt.hlines(1, 0, 500)
    plt.plot(a/10, "r")
    plt.plot(b/10, "g")

    plt.xlim(0, 500)
    plt.ylim(0, 1)


    plt.show()