import itertools
import csv
import time
import os
import sys

from typing import List

from .losses import *
from .regularizers import *

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, Manager

keys_training_params = ["epochs", "batch_size", "early_stopping"] 
keys_training_params_direct = ["lambda_", "p_d", "p_dc"]


def save_results_to_file(keys, gs_results, path):
    timestr = time.strftime("%Y-%m-%d-%H%M%S")
    path = os.path.splitext(path)[0]

    with open(path+"-"+timestr+".csv", 'w') as f:
        writer = csv.writer(f)

        writer.writerow(keys + ["loss_tr", "loss_vl"])

        for (params, loss_tr, loss_vl) in gs_results:
            writer.writerow(list(params.values()) + [loss_tr, loss_vl])


def grid_parallel(shared_queue, model, train_data, valid_data, direct, training_params, search_param):
    if not direct:
        history = model.training(train_data, valid_data, **training_params)
        shared_queue.put((search_param, history["loss_tr"][-1], history["loss_vl"][-1]))
    else:
        loss_tr, loss_vl = model.direct_training(train_data, valid_data, **training_params)
        shared_queue.put((search_param, loss_tr, loss_vl))
    print(search_param, "  : done!!")


def grid_parallel_cv(shared_queue, build_model, dataset, k_folds, direct, static_params, search_param):
    loss_tr_cv, loss_vl_cv = cross_validation(build_model, dataset, {**static_params, **search_param}, k_folds, direct)
    shared_queue.put((search_param, loss_tr_cv, loss_vl_cv))
    print(search_param, " : done!!")


def best_comb(gs_results):
    best_result = np.inf
    best_combination = None
    for (search_param, loss_tr, loss_vl) in gs_results:

        print(f"{search_param} -> tr: {loss_tr} | vl: {loss_vl}")

        if best_result > loss_vl:
           best_result = loss_vl
           best_combination = search_param

    return best_combination


def split_train_params(params, direct=False):
    """
    Splitting training parameters with epochs and batch_size
    params = parameters
    """

    if not direct:
        keys_training = keys_training_params
    else:
        keys_training = keys_training_params_direct

    other_params = { key:value for key,value in params.items() if key not in keys_training}
    training_params = { key:value for key,value in params.items() if key in keys_training}
    return other_params, training_params


def split_search_params(params):
    """
    Splitting search parameters with static parameters (i.e. parameters like epochs or batch_size)
    """
    static_params = { key:value for key,value in params.items() if not isinstance(value, List) }
    search_params = { key:value for key,value in params.items() if isinstance(value, List)}
    return static_params, search_params


def cross_validation(build_model, dataset: tuple, params:dict, k_folds=4, direct=False):
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

    build_params, train_params = split_train_params(params, direct)

    loss_tr_mean = 0
    loss_vl_mean = 0

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

        if not direct:
            history = model.training((train_x, train_y), (valid_x, valid_y), **train_params)
            loss_tr_mean += history["loss_tr"][-1]
            loss_vl_mean += history["loss_vl"][-1]
        else:
            loss_tr, loss_vl = model.direct_training((train_x, train_y), (valid_x, valid_y), **train_params)
            loss_tr_mean += loss_tr
            loss_vl_mean += loss_vl

    loss_tr_mean /= k_folds
    loss_vl_mean /= k_folds

    return loss_tr_mean, loss_vl_mean


def grid_search_cv(build_model, dataset, params:dict, k_folds=4, direct=False, path=None):
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
    
    try:

        for param_combination in itertools.product(*search_params.values()):
            # create dictionary for params
            search_param = {}
            for i, param_key in enumerate(search_params.keys()):
                search_param[param_key] = param_combination[i]

            print("-> ", search_param)

            # here i have data to pass to the workers
            pool.apply_async(grid_parallel_cv, (shared_queue, build_model, dataset, k_folds, direct, static_params, search_param))
        
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        sys.exit(1)

    gs_results = []
    while not shared_queue.empty():
        gs_results.append(shared_queue.get())

    # salvare i risultati della grid search in un file se presente il path
    if path:
        save_results_to_file(list(search_params.keys()), gs_results, path)

    return_data = {**best_comb(gs_results), **static_params}

    return return_data


# drop to the build_model the task to assign the params to build the model
def grid_search(build_model, train_data, valid_data, params:dict, direct=False, path=None):
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
    for param_combination in itertools.product(*search_params.values()):
        # create dictionary for params
        search_param = {}
        for i, param_key in enumerate(search_params.keys()):
            search_param[param_key] = param_combination[i]

        print("-> ", search_param)
                
        build_params, training_params = split_train_params({**static_params, **search_param}, direct)

        model = build_model(**build_params)

        # here i have data to pass to the workers
        pool.apply_async(grid_parallel, args=(shared_queue, model, train_data, valid_data, direct, training_params, search_param))

    pool.close()
    pool.join()

    gs_results = []
    while not shared_queue.empty():
        gs_results.append(shared_queue.get())

    # salvare i risultati della grid search in un file se presente il path
    if path:
        save_results_to_file(list(search_params.keys()), gs_results, path)


    return_data = {**best_comb(gs_results), **static_params}

    return return_data