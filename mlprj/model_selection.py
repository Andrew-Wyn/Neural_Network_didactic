import itertools
import math

from typing import List

from .losses import *
from .regularizers import *

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, Manager

from functools import partial


keys_training_params = ["epochs", "batch_size"] 


def grid_parallel(shared_queue, model, train_data, valid_data, training_params, search_param):
    history = model.training(train_data, valid_data, **training_params)
    shared_queue.put((search_param, history["loss_tr"], history["loss_vl"]))
    print(search_param, "  : done!!")


def grid_parallel_cv(build_model, shared_queue, dataset, static_params, search_param):
    loss_tr_cv, loss_vl_cv = cross_validation(build_model, dataset, {**static_params, **search_param})
    shared_queue.put((search_param, loss_tr_cv, loss_vl_cv))
    print(search_param, " : done!!")


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
    """
    Splitting training parameters with epochs and batch_size
    params = parameters
    """
    other_params = { key:value for key,value in params.items() if key not in keys_training_params}
    training_params = { key:value for key,value in params.items() if key in keys_training_params}
    return other_params, training_params

def split_search_params(params):
    """
    Splitting search parameters with static parameters (i.e. parameters like epochs or batch_size)
    """
    static_params = { key:value for key,value in params.items() if not isinstance(value, List) }
    search_params = { key:value for key,value in params.items() if isinstance(value, List)}
    return static_params, search_params


def cross_validation(build_model, dataset: tuple, params:dict, k_folds=4):
    """
    Perform a k-fold cross-validation with k = k_folds.
    build_model = model architecture
    dataset = data set
    params = dictionary of the parameters
    k_folds = number of folds of cross validation (i.e. k_folds = k)
    """
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
    """
    Perform a grid search in which for every n-uple of parameters we use a 4-fold cross validation
    for a better estimate of training and validation error.
    build_model = model architecture
    dataset = data set
    params = dictionary of parameters 
    """
    static_params, search_params = split_search_params(params)
    m = Manager()
    shared_queue = m.Queue()

    pool = Pool() # use all available cores, otherwise specify the number you want as an argument
    for param_combination in itertools.product(*search_params.values()):
        # create dictionary for params
        search_param = {}
        for i, param_key in enumerate(search_params.keys()):
            search_param[param_key] = param_combination[i]

        print("-> ", search_param)

        # here i have data to pass to the workers
        pool.apply_async(partial(grid_parallel_cv, build_model), (shared_queue, dataset, static_params, search_param))
        
    pool.close()
    pool.join()

    gs_results = []
    while not shared_queue.empty():
        gs_results.append(shared_queue.get())

    fig, spec, num_cols, _ = make_plot_grid(search_params)
    best_result = np.inf
    best_combination = None
    for j, (search_param, loss_tr, loss_vl)  in enumerate(gs_results):
        col = j%num_cols
        row = j//num_cols

        ax = fig.add_subplot(spec[row, col])
        ax.title.set_text(search_param)
        
        ax.plot(loss_tr)
        ax.plot(loss_vl)

        result = loss_vl[-1]

        if best_result > result:
           best_result = result
           best_combination = search_param

    plt.show()
    plt.clf()

    return best_combination


# drop to the build_model the task to assign the params to build the model
def grid_search(build_model, train_data, valid_data, params:dict):
    """
    TODO: scrivere documentazione seguendo standard pep8
    Perform a classic grid_search.
    train_data = training data set
    valid_data = validation data set
    build_model = model architecture
    params = dictionary of parameters
    """

    static_params, search_params = split_search_params(params)
    m = Manager()
    shared_queue = m.Queue()

    pool = Pool() # use all available cores, otherwise specify the number you want as an argument
    for j, param_combination in enumerate(itertools.product(*search_params.values())):
        # create dictionary for params
        search_param = {}
        for i, param_key in enumerate(search_params.keys()):
            search_param[param_key] = param_combination[i]

        print("-> ", search_param)
                
        build_params, training_params = split_train_params({**static_params, **search_param})

        model = build_model(**build_params)

        # here i have data to pass to the workers
    
        pool.apply_async(grid_parallel, args=(shared_queue, model, train_data, valid_data, training_params, search_param))


    pool.close()
    pool.join()

    gs_results = []
    while not shared_queue.empty():
        gs_results.append(shared_queue.get())
    
    print(len(gs_results))

    fig, spec, num_cols, _ = make_plot_grid(search_params)
    best_result = np.inf
    best_combination = None
    for j, (search_param, loss_tr, loss_vl)  in enumerate(gs_results):
        col = j%num_cols
        row = j//num_cols

        ax = fig.add_subplot(spec[row, col])
        ax.title.set_text(search_param)
        
        ax.plot(loss_tr)
        ax.plot(loss_vl)

        result = loss_vl[-1]

        if best_result > result:
           best_result = result
           best_combination = search_param

    plt.show()
    plt.clf()

    return best_combination
        