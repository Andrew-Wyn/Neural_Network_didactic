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
    def __init__(self, input_dim: int, layers=[]):
      self.input_dim = input_dim
      self.last_layer = layers[-1]
      self.layers = layers[:-1]

      # handled by compile
      self.loss = None
      self.regularizer = None
    
    def compile(self, loss=None, regularizer=None, optimizer=None):
        """
        Function that inizializes the neural network
        Args:
            loss: selected loss function
            regularizer: selected regularization method
            optimizer: optimization procedure
        """
        self.loss = loss
        self.regularizer = regularizer
        self.optimizer = optimizer

        prev_dim = self.input_dim 
        for layer in self.layers:
            layer.initialize_weights(prev_dim)
            prev_dim = layer.output_dim

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
        for layer in self.layers:
            value = layer.forward_step(value)
        return self.last_layer.forward_step(value)

    def predict(self, inputs):
      preds = []

      for net_input in inputs:
        preds.append(self.forward_step(net_input))

      return np.array(preds)

    def _bp_delta_weights(self, net_input: np.array, target: np.array):
        # calculate the delta of the last layer

        fw_out = self.forward_step(net_input)

        dE_dO = self.loss.derivate(target, fw_out)
        gradient_w, gradient_b = self.last_layer.calculate_gradient(dE_dO)
        
        return -gradient_w, -gradient_b

    def _apply_deltas(self, deltas):
        delta_w, delta_b = deltas
        self.last_layer.weights_matrix += delta_w
        self.last_layer.bias += delta_b 

    def _regularize(self):
        """
        Function that returns the regularization part of the deltaW update rule
        Args:
            regs: (list) weights and biases regularization part
        """
        reg_w = self.regularizer.regularize(self.last_layer.weights_matrix)
        reg_b = self.regularizer.regularize(self.last_layer.bias)

        return reg_w, reg_b

    def direct_training(self, training, validation=None, lambda_=0, verbose=False):
      input_tr, target_tr = training

      valid_split = False

      if validation:
        input_vl, target_vl = validation
        valid_split = True

      # transform input data to high dim rand layer
      transformed_train = []
      for sample in input_tr:
        transformed_sample = sample
        for layer in self.layers:
            transformed_sample = layer.forward_step(transformed_sample)
        transformed_train.append(transformed_sample)

      transformed_train = np.array(transformed_train)

      # output_weights = np.dot(pinv(transformed_train), target_tr)

      hidden_layer_dim = self.layers[-1].output_dim

      output_weights = np.linalg.lstsq(transformed_train.T.dot(transformed_train) + lambda_ * np.identity(hidden_layer_dim), transformed_train.T.dot(target_tr), rcond=None)[0]

      self.last_layer.weights_matrix = output_weights.T
      self.last_layer.bias = np.zeros(self.last_layer.bias.shape)

      error_tr = self.compute_total_error(input_tr, target_tr)
      error_vl = -1

      if valid_split:
        error_vl = self.compute_total_error(input_vl, target_vl)
        if verbose:
          print(f"error_tr = {error_tr} | error_vl = {error_vl}")
      else:
        if verbose:
          print(f"error_tr = {error_tr}")

      return error_tr, error_vl

    def training(self, training, validation=None, epochs=500, batch_size=64, verbose=False):
      """
        Function that performs neural network training phase, choosing the minibatch size if needed
        Args:
            training: training set
            validation: validation set
            epochs: number of epochs
            batch_size: batch size
        Returns:
            history: training and validation error for each epoch
      """
            
      input_tr, target_tr = training
      if validation is not None:
        input_vl, target_vl = validation
        valid_split = True
      else:
        valid_split = False
      
      history = {"loss_tr": [], "loss_vl": []}

      l = len(input_tr)

      if type(batch_size) is str and batch_size == "full":
        batch_size = l
      elif type(batch_size) is str and batch_size != "full":
        raise Exception()
      elif batch_size > l or batch_size <= 0:
        batch_size = l

      for i in range(epochs):

        for batch_number in range(math.ceil(l//batch_size)):
          # sum deltas over the batch
          deltas = None

          batch_iterations = 0
          for batch_i in range(batch_number*batch_size, min(batch_number*batch_size + batch_size, l)):
            batch_iterations += 1

            x = input_tr[batch_i]
            y = target_tr[batch_i]

            if not deltas:
                deltas = self._bp_delta_weights(x, y)
            else:
                update_delta_w, update_delta_b = self._bp_delta_weights(x, y)
                delta_w, delta_b = deltas
                deltas = (delta_w + update_delta_w, delta_b + update_delta_b)

          # at this point we have delta summed-up to all batch instances
          # let's average it
          # sumgrad/mb * (mb/l) * eta
          delta_w, delta_b = deltas
          deltas = (delta_w/l, delta_b/l)
          
          regs = self._regularize()

          # calculate the delta throught the optimizer
          optimized_deltas = self.optimizer.optimize([deltas], [regs], i)

          self._apply_deltas(optimized_deltas[0])

        epoch_error_tr = self.compute_total_error(input_tr, target_tr)
        history["loss_tr"].append(epoch_error_tr)

        if valid_split:
          epoch_error_vl = self.compute_total_error(input_vl, target_vl)
          history["loss_vl"].append(epoch_error_vl)
          if verbose:
            print(f"epoch {i}: error_tr = {epoch_error_tr} | error_vl = {epoch_error_vl}")
        else:
          if verbose:
            print(f"epoch {i}: error_tr = {epoch_error_tr}")

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

    def calculate_gradient(self, output_dE_dO):
        dE_dNet = output_dE_dO * self.activation_derivate(self._net)
        gradient_w = np.transpose(dE_dNet[np.newaxis, :]) @ self._input[np.newaxis, :] 
        gradient_b = dE_dNet

        return gradient_w, gradient_b