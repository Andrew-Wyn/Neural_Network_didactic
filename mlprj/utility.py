import numpy as np

def model_accuracy(model, X, y, threshold=0.5):
    preds = model.predict(X) >= threshold
    return 1 - np.mean(np.abs(preds - y), axis=0)


def model_loss(model, loss, X, y):
    preds = model.predict(X)
    total_error = 0
    for pred, target in zip(preds, y):
        total_error += loss.compute(pred, target)
    return total_error/len(X)
