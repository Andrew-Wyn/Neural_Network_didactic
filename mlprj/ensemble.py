import numpy as np


class Ensemble():
    """
    Ensemble model
    """

    def __init__(self, models=[]):
        """
        Args:
            models: (list) list of models
        """

        self.models = models

    def add_model(self, model):
        """
        Add a model to the ensemble

        Args:
            model: a model to be added to the ensemble
        """

        self.models.append(model)
    
    def predict(self, X):
        """
        Perform a prediction over the entire models of the ensemble and compute the mean
        
        Args:
            X: (np.ndarray) list of inputs
        Returns:
            means: (np.ndarray) the mean over the axis 0 of the predicted vector
        """
        
        means = np.mean(np.array([model.predict(X) for model in self.models]), axis = 0)
        return means