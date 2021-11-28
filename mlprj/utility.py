import numpy as np

def model_accuracy(model, X, y):
    preds = model.predict(X) >= 0.5
    return 1 - np.mean(np.abs(preds - y), axis=0)