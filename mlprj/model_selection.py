import itertools
import math

from typing import List


from losses import *
from regularizers import *

import numpy as np
import matplotlib.pyplot as plt


keys_training_params = ["epochs", "batch_size"] 

def make_plot_grid(search_params):
    num_params = len(search_params)

    num_cols = 1
    num_rows = 1
    for i, x in enumerate(search_params.values()):
        if i+1 > math.ceil(num_params/2):
            num_rows *= len(x)
        else:
            num_cols *= len(x)

    fig = plt.figure(constrained_layout=True, figsize=(3*num_cols,3*num_rows))
    spec = fig.add_gridspec(ncols=num_cols, nrows=num_rows)
    return fig, spec, num_cols, num_rows


def split_train_params(params):
    other_params = { key:value for key,value in params.items() if key not in keys_training_params}
    training_params = { key:value for key,value in params.items() if key in keys_training_params}
    return other_params, training_params

def split_search_params(params):
    static_params = { key:value for key,value in params.items() if not isinstance(value, List) }
    search_params = { key:value for key,value in params.items() if isinstance(value, List)}
    return static_params, search_params


def cross_validation(build_model, dataset: tuple, params:dict, k_folds=4):
    X, y = dataset

    l = len(X)
    l_vl = l//k_folds

    build_params, train_params = split_train_params(params)

    loss_tr_mean = np.zeros(train_params["epochs"])
    loss_vl_mean = np.zeros(train_params["epochs"])

    # TODO: possibilità di utilizzare KFOLD di sklearn (?)
    for k in range(k_folds):
        if k != k_folds-1:
            train_x = np.append(X[:(k)*l_vl], X[(k+1)*l_vl:], axis=0)
            train_y = np.append(y[:(k)*l_vl], y[(k+1)*l_vl:], axis=0)
            valid_x = X[k*l_vl:(k+1)*l_vl]
            valid_y = y[k*l_vl:(k+1)*l_vl]
        else: # last fold clausole
            train_x = X[:k*l_vl]
            train_y = y[:k*l_vl]
            valid_x = X[k*l_vl:]
            valid_y = y[k*l_vl:]

        model = build_model(**build_params)
        history = model.training((train_x, train_y), (valid_x, valid_y), **train_params)

        loss_tr_mean += history["loss_tr"]
        loss_vl_mean += history["loss_vl"]

    loss_tr_mean /= k_folds
    loss_vl_mean /= k_folds

    return loss_tr_mean, loss_vl_mean


def grid_search_cv(build_model, dataset, params:dict):

    static_params, search_params = split_search_params(params)

    fig, spec, num_cols, num_rows = make_plot_grid(search_params)

    best_result = np.inf
    best_combination = None
    for j, param_combination in enumerate(itertools.product(*search_params.values())):
        # create dictionary for params
        search_param = {}
        for i, param_key in enumerate(search_params.keys()):
            search_param[param_key] = param_combination[i]

        print("-> ", search_param)

        loss_tr_cv, loss_vl_cv = cross_validation(build_model, dataset, {**static_params, **search_param})

        col = j%num_cols
        row = j//num_cols

        print(row, col)

        ax = fig.add_subplot(spec[row, col])
        
        ax.plot(loss_tr_cv)
        ax.plot(loss_vl_cv)

        result = min(loss_vl_cv)

        if best_result > result:
            best_result = result
            best_combination = param_combination

    # create dictionary for best params
    best_param = {}
    for i, param_key in enumerate(search_params.keys()):
        best_param[param_key] = best_combination[i]

    plt.show()
    plt.clf()

    return best_param


# drop to the build_model the task to assign the params to build the model
def grid_search(model, train_data, valid_data, build_model, params:dict):
    """
    search_params {"par_1": parameters, "par_2": parameters, ...}
    """

    static_params, search_params = split_search_params(params)

    best_result = np.inf
    best_combination = None
    for param_combination in itertools.product(*search_params.values()):
        # create dictionary for params
        search_param = {}
        for i, param_key in enumerate(search_params.keys()):
            search_param[param_key] = param_combination[i]

        print("-> ", search_param)
        
        build_params, training_params = split_train_params({**static_params, **search_param})
        
        model = build_model(model, **build_params)
        history = model.training(train_data, valid_data, **training_params)
        
        result = min(history["loss_vl"])

        print(result)

        if best_result > result:
            best_result = result
            best_combination = param_combination
    
    # create dictionary for best params
    best_param = {}
    for i, param_key in enumerate(search_params.keys()):
        best_param[param_key] = best_combination[i]
    
    return best_param
        