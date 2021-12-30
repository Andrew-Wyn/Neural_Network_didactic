import numpy as np


class Ensemble():
    """
        Ensemble model
    """
    def __init__(self, models=[]):
        self.models = models

    def add_model(self, model):
        self.models.append(model)
    
    def predict(self, X):
        return np.mean(np.array([model.predict(X) for model in self.models]), axis = 0)