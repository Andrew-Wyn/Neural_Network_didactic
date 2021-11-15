import numpy as np

from activations import *
from initializations import *


class Layer:
  """
      Class of the Layer
      Attributes:
          output_dim: (int) that indicate the dimension of the output
          activation: (str) name of the activation function
          initialization: (str) name of the initialization procedure
          initialization_parameters: (dict) hyperparameters
  """
  def __init__(self, previous_layers, output_dim, activation, initialization, initialization_parameters={}):
        print(previous_layers)
        self.input_dim = sum([
          len(layer.output_dim) for layer in previous_layers
        ])
        self.output_dim = output_dim 
        self.weights_matrix = None
        self.bias = None
        self.activation = activation_functions[activation]
        self.activation_derivate = derivate_activation_functions[activation]

        self.initialization = lambda input_dim, output_dim: initialization_functions[initialization](input_dim, output_dim, **initialization_parameters)

        self._input = None
        self._net = None
        self._output = None

        self.previous_layers = previous_layers

  def initialize_weights(self, input_dim):
    """
      Function that initializes the weights
      Args:
          input_dim: (int) the dimension of the input
    """
    self.input_dim += input_dim
    self.weights_matrix, self.bias = self.initialization(self.input_dim, self.output_dim)

  def forward_step(self, input: np.ndarray):
    """
      Function that implement a layer forward step
      Args:
          input: (np.array) initial net values
      Returns:
          output: layer's output
    """
    internal_outputs = []

    for layer in self.previous_layers:
      np.append(internal_outputs, input)

    self._input = np.append(internal_outputs, input)
    self._net = np.matmul(self.weights_matrix, self._input)
    self._net = np.add(self._net, self.bias)
    self._output = self.activation(self._net)
    return self._output


class CascadeCorrelation:
  def __init__(self, input_dim, output_dim, activation, initialization, initialization_parameters={}):
    # last neuron stuffs
    self.input_dim = input_dim # fixed input dimension
    self._real_input_dim = None # dimension of the real input of the out
    self.output_dim = output_dim
    self.activation = activation_functions[activation]
    self.activation_derivate = derivate_activation_functions[activation]
    self.initialization = lambda input_dim, output_dim: initialization_functions[initialization](input_dim, output_dim, **initialization_parameters)
    self.weights_matrix = None
    self.bias = None
    self._input = None
    self._net = None
    self._output = None

    # entire network stuffs
    self.layers = []

  def compile(self, loss=None, regularizer=None, optimizer=None):
    self.loss = loss
    self.regularizer = regularizer
    self.optimizer = optimizer
    self.refresh()

  def refresh(self):
    # necessary when a new layer is added to the network
    # calculate the real input dim
    self._real_input_dim = sum([
      layer.output_dim for layer in self.layers
    ]) + self.input_dim
    
    # init output layer weights
    self.weights_matrix, self.bias = self.initialization(self._real_input_dim, self.output_dim)

  def forward_step(self, input: np.array):
    internal_outputs = []

    for layer in self.layers:
      internal_outputs.append(layer.forward_step(np.append(internal_outputs, input)))

    self._input = np.append(internal_outputs, input)

    self._net = np.matmul(self.weights_matrix, self._input)
    self._net = np.add(self._net, self.bias)
    self._output = self.activation(self._net)
    return self._output


  def _perceptron_delta_weights(self, input: np.array, target: np.array): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
    """
    """

    fw_out = self.forward_step(input)
    
    dE_dO = self.loss.derivate(target, fw_out)
    dE_dNet = dE_dO * self.activation_derivate(self._net)
    gradient_w = np.transpose(dE_dNet[np.newaxis, :]) @ self._input[np.newaxis, :] 
    gradient_b = dE_dNet
    
    return (-gradient_w, -gradient_b)

  def _correlation_delta_weights(self, Vs, Es, Ds, Is): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
    """
      Vs = np.array represent the candidate neuron outputs
      Es = np.array represent the error of the output and the target
      Ds = np.array represent the derivatives of activation function of the output layer
      Is = np.array represent the inputs of the candidate unit
    """
    
    Vmean = np.mean(Vs, axis=0)
    Emean = np.mean(Es, axis=0)

    # lenght of sigma_o equal to the output dim of the network
    sigma = np.sign((Vs - Vmean).T@(Es - Emean))

    gradient_b = ((sigma*(Es - Emean)).T@Ds)
    gradient_w = (sigma*(Es - Emean)).T@(Ds*Is) # lenght I

    return (gradient_w, gradient_b)
    
  def _regularize(self):
    reg_w = self.regularizer.regularize(self.weights_matrix)
    reg_b = self.regularizer.regularize(self.bias)
    return (reg_w, reg_b)


  def training(self, training, validation, epochs=500, batch_size=64):
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
    input_vl, target_vl = validation
    
    history = {"loss_tr": [], "loss_vl": []}

    l = len(input_tr)

    if type(batch_size) is str and batch_size == "full":
      batch_size = l

    for i in range(epochs): # per ogni epoca aggiungo un neurone

      # alleno pesi collegati con l'output e calcolo l'errore commesso alla fine dell'allenamento
      for batch_number in range(l//batch_size):
        delta = None

        batch_iterations = 0
        for batch_i in range(batch_number*batch_size, min(batch_number*batch_size + batch_size, l)):
          batch_iterations += 1

          x = input_tr[batch_i]
          y = target_tr[batch_i]

          if not delta:
            delta = self._perceptron_delta_weights(x, y)
          else:
            update_w, update_b = self._perceptron_delta_weights(x, y)
            delta_w, delta_b = delta
            delta = (delta_w + update_w, delta_b + update_b)

        # at this point we have delta summed-up to all batch instances
        # let's average it
        delta_w, delta_b = delta
        delta = (delta_w/batch_iterations, delta_b/batch_iterations)
        
        regs = self._regularize()
        # calculate the delta throught the optimizer
        optimized_deltas = self.optimizer.optimize([delta], [regs])[0]

        delta_w, delta_b = optimized_deltas
        self.weights_matrix += delta_w
        self.bias += delta_b

      epoch_error_tr = self.compute_total_error(input_tr, target_tr)
      epoch_error_vl = self.compute_total_error(input_vl, target_vl)

      print(f"epoch {i}: error_tr = {epoch_error_tr} | error_vl = {epoch_error_vl}")
      history["loss_tr"].append(epoch_error_tr)
      history["loss_vl"].append(epoch_error_vl)

      if i < epochs-1:
        # aggiungo un hidden unit, i cui pesi sono calcolati tramite covarianza
        new_layer = Layer(self.layers, 1, "relu", "gaussian")

        new_layer.initialize_weights(self.input_dim)

        # covariance learning
        for batch_number in range(l//batch_size):
          Vs = []
          Es = []
          Ds = []
          Is = []

          batch_iterations = 0
          for batch_i in range(batch_number*batch_size, min(batch_number*batch_size + batch_size, l)):
            batch_iterations += 1

            x = input_tr[batch_i]
            y = target_tr[batch_i]

            O_p = self.forward_step(x)
            E_p = O_p - y
            Es.append(E_p)
            # calculate output of the new hidden unit
            # TODO: move inside layer
            Vs.append(new_layer.forward_step(x))
            Is.append(new_layer._input)
            Ds.append(new_layer.activation_derivate(new_layer._net))

          # at this point we have delta summed-up to all batch instances

          # move inside the optimizator          
          S_p_w, S_p_b = self._correlation_delta_weights(np.array(Vs), np.array(Es), np.array(Ds), np.array(Is))     
          new_layer.weights_matrix += 0.1*S_p_w
          new_layer.bias += 0.1*S_p_b[0]

        self.layers.append(
          new_layer # to parametrize outside
        )

        self.refresh()

    return history

  def compute_total_error(self, input, target):
    """
      The computation of the total error
      Args:
          input: (np.array) initial net values
          target: the target value
      Returns:
          output: total error
    """
    total_error = 0
    for i in range(len(input)):
      total_error += self.loss.compute(target[i], self.forward_step(input[i]))
    return total_error/len(input)
