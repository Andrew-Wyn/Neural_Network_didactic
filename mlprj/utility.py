import numpy as np

def model_accuracy(model, X, y):
    preds = model.predict(X) >= 0.5
    return 1 - np.mean(np.abs(preds - y), axis=0)

def model_loss(model, loss, X, y):
    total_error = 0
    for i in range(len(X)):
        total_error += loss.compute(y[i], model.forward_step(X[i]))
    return total_error/len(X)
