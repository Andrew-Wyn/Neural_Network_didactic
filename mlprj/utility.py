import numpy as np


def compiled_check(method):
    """
    Decorator used to be applied to the method of the class Network and RandomNetwork
    the method decorated can be executed if the model is compiled
    """
    def inner(obj, *args, **kwargs):
        if not obj.compiled:
            raise RuntimeError("model not compiled")
        return method(obj, *args, **kwargs)

    return inner


def model_accuracy(model, X, y, threshold=0.5):
    """
    Compute the model accuracy given a threshold

    Args:
        model: the model
        X: (np.ndarray) dataset
        y: (np.ndarray) target
        threshold: (double)
    """
    preds = model.predict(X) >= threshold
    return np.mean(1 - np.mean(np.abs(preds - y), axis=0))


def model_loss(model, loss, X, y):
    """
    Compute the model loss given a loss function

    Args:
        model: the model
        loss: (Loss) loss function
        X: (np.ndarray) dataset
        y: (np.ndarray) target
    """
    preds = model.predict(X)
    total_error = 0
    for pred, target in zip(preds, y):
        total_error += loss.compute(pred, target)
    return total_error/len(X)
