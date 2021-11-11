import itertools

from losses import *
from regularizers import *

import numpy as np
import matplotlib.pyplot as plt


keys_training_params = ["epochs", "batch_size"] 


def split_train_params(**total_params):
    other_params = { key:value for key,value in total_params.items() if key not in keys_training_params}
    training_params = { key:value for key,value in total_params.items() if key in keys_training_params}
    return other_params, training_params


def grid_search_cv(model, train_data, fixed_params:dict, search_params:dict):
    pass


# drop to the build_model the task to assign the params to build the model
def grid_search(model, train_data, valid_data, build_model, search_params:dict):
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

        print("-> ", search_param)
        
        others_params, training_params = split_train_params(**search_param)
        
        model = build_model(model, **others_params)
        history = model.training(train_data, valid_data, **training_params)
        
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
        