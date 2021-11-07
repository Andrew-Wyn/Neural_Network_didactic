import itertools

from losses import *
from regularizers import *


import numpy as np
import matplotlib.pyplot as plt

def prepare_and_train_model(model, train_data, valid_data, loss, regularizer, fixed_params: dict, search_param:dict):
    
    model.compile(loss=loss, regularizer=regularizer)

    return model.training(train_data, valid_data, **fixed_params, **search_param)

def grid_search(model, train_data, valid_data, loss, regularizer, fixed_params: dict, search_params:dict):
    """
    search_params {"par_1": parameters, "par_2": parameters, ...}
    """

    results = []
    best_result = np.inf
    best_combination = None
    for param_combination in itertools.product(*search_params.values()):
        # create dictionary for params
        search_param = {}
        for i, param_key in enumerate(search_params.keys()):
            search_param[param_key] = param_combination[i]

        print(search_param)
        
        history = prepare_and_train_model(model, train_data, valid_data, loss, regularizer, fixed_params, search_param)
        
        results.append(history["loss_vl"])

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
        