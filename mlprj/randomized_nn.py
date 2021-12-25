import numpy as np
import math

from scipy.linalg import pinv

from .activations import *
from .initializers import *

class RandomizedNetwork:
    """
        Class of the neural network
        Attributes:
            input_dim: (int) the dimension of the input
            layers: (list) the layers
            loss: the loss function
            regularizer: the regularization parameter
            optimizer: learning rate and Polnjak momentum parameters

    """
    def __init__(self, input_dim: int, hidden_layer, output_dim):
      self.input_dim = input_dim
      self.last_layer = last_layer
      self.hidden_layer = hidden_layer

      # handled by compile
      self.loss = None
    
    def compile(self, loss=None):
        """
        Function that inizializes the neural network
        Args:
            loss: selected loss function
            regularizer: selected regularization method
            optimizer: optimization procedure
        """
        self.loss = loss

        prev_dim = self.input_dim 
        self.hidden_layer.initialize_weights(prev_dim)
        prev_dim = self.hidden_layer.output_dim
        self.last_layer.initialize_weights(prev_dim)
    
    def forward_step(self, net_input: np.ndarray):
        """
        A forward step of the network
        Args:
            net_input: (np.array) of initial net values
        Returns:
            value: network's output
        """
        value = net_input
        value = self.hidden_layer.forward_step(value)
        return self.last_layer.forward_step(value)

    def predict(self, inputs):
      preds = []

      for net_input in inputs:
        preds.append(self.forward_step(net_input))

      return np.array(preds)

    def bias_dropout(self, H, p_d, quantile=0.5):
      mask = np.full(H.shape, 1)

      threshold = np.quantile(H, quantile)

      locs = H < threshold

      subs = np.random.binomial(1, 1-p_d, np.sum(locs))

      mask[locs] = subs

      return np.multiply(H, mask)

    def direct_training(self, training, validation=None, lambda_=0, p_d=0, p_dc=0, verbose=False):
      input_tr, target_tr = training

      valid_split = False

      history = {"loss_tr": None, "loss_vl": -1}

      if validation:
        input_vl, target_vl = validation
        valid_split = True

      # Biased DropConnect
      self.hidden_layer.drop_connect(p_dc)

      # transform input data to high dim rand layer
      transformed_train = []
      for sample in input_tr:
        transformed_sample = sample
        for layer in self.layers:
            transformed_sample = layer.forward_step(transformed_sample)
        transformed_train.append(transformed_sample)

      transformed_train = np.array(transformed_train)

      # Biased Dropout
      transformed_train = self.bias_dropout(transformed_train, p_d)

      # un-regularized learner
      # output_weights = np.dot(pinv(transformed_train), target_tr)

      # regularized learner
      hidden_layer_dim = self.hidden_layer.output_dim
      output_weights = np.linalg.lstsq(transformed_train.T.dot(transformed_train) + lambda_ * np.identity(hidden_layer_dim), transformed_train.T.dot(target_tr), rcond=None)[0]

      self.last_layer.weights_matrix = output_weights.T
      self.last_layer.bias = np.zeros(self.last_layer.bias.shape)

      error_tr = self.compute_total_error(input_tr, target_tr)

      history["loss_tr"] = error_tr

      if valid_split:
        error_vl = self.compute_total_error(input_vl, target_vl)
        if verbose:
          print(f"error_tr = {error_tr} | error_vl = {error_vl}")
        history["loss_vl"] = error_vl
      else:
        if verbose:
          print(f"error_tr = {error_tr}")

      return history

    def compute_total_error(self, net_input, target):
      """
        The computation of the total error
        Args:
            input: (np.array) initial net values
            target: the target value
        Returns:
            output: total error
      """
      total_error = 0
      for i in range(len(net_input)):
        total_error += self.loss.compute(target[i], self.forward_step(net_input[i]))
      return total_error/len(net_input)


class RandomizedLayer:
    """
        Class of the Layer
        Attributes:
            output_dim: (int) that indicate the dimension of the output
            activation: (str) name of the activation function
            initialization: (str) name of the initialization procedure
            initialization_parameters: (dict) hyperparameters
    """
    def __init__(self, output_dim, activation="relu", initializer=GaussianInitializer()):
        self.input_dim = None # if it's not setted, not "compiled" the nn
        self.output_dim = output_dim
        self.weights_matrix = None
        self.bias = None
        self.activation = activation_functions[activation]
        self.activation_derivate = derivate_activation_functions[activation]

        self.initializer = initializer

        self._input = None
        self._net = None
        self._output = None

    def initialize_weights(self, input_dim):
        """
        Function that initializes the weights
        Args:
            input_dim: (int) the dimension of the input
        """
        self.input_dim = input_dim
        self.weights_matrix, self.bias = self.initializer.initialize(self.input_dim, self.output_dim)

    def forward_step(self, net_input: np.ndarray):
        """
        Function that implement a layer forward step
        Args:
            net_input: (np.array) initial net values
        Returns:
            output: layer's output
        """
        self._input = net_input
        self._net = np.matmul(self.weights_matrix, self._input)
        self._net = np.add(self._net, self.bias)
        self._output = self.activation(self._net)
        return self._output

    def drop_connect(self, p_dc, quantile=0.5):
      mask_w = np.full(self.weights_matrix.shape, 1)
      mask_b = np.full(self.bias.shape, 1)

      threshold_w = np.quantile(self.weights_matrix, quantile)
      threshold_b = np.quantile(self.bias, quantile)

      locs_w = self.weights_matrix < threshold_w
      locs_b = self.bias < threshold_b

      subs_w = np.random.binomial(1, 1-p_dc, np.sum(locs_w))
      subs_b = np.random.binomial(1, 1-p_dc, np.sum(locs_b))

      mask_w[locs_w] = subs_w
      mask_b[locs_b] = subs_b

      self.weights_matrix = np.multiply(self.weights_matrix, mask_w)
      self.bias = np.multiply(self.bias, mask_b)
