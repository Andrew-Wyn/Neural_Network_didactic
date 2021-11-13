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
  def __init__(self, output_dim, activation, initialization, initialization_parameters={}):
      self.input_dim = None # if it's not setted, not "compiled" the nn
      self.output_dim = output_dim 
      self.weights_matrix = None
      self.bias = None
      self.activation = activation_functions[activation]
      self.activation_derivate = derivate_activation_functions[activation]

      self.initialization = lambda input_dim, output_dim: initialization_functions[initialization](input_dim, output_dim, **initialization_parameters)

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
    self.weights_matrix, self.bias = self.initialization(self.input_dim, self.output_dim)

  def forward_step(self, input: np.ndarray):
    """
      Function that implement a layer forward step
      Args:
          input: (np.array) initial net values
      Returns:
          output: layer's output
    """
    self._input = input
    self._net = np.matmul(self.weights_matrix, self._input)
    self._net = np.add(self._net, self.bias)
    self._output = self.activation(self._net)
    return self._output

  def backpropagate_delta(self, upper_dE_dO): # return current layer
    # calculate dE_dNet for every node in the layer = dE_dO * fprime(net)
    """
      Function that implement backpropagation algorithm
      Args:
          upper_dE_dO: derivative of E with respect to the output O for the upper layer
      Returns:
          current dE_do: the current derivative of E with respect to O,
          gradient_w: the gradient with respect to W
          gradient_b: the gradient with respect to the bias
    """
    dE_dNet = upper_dE_dO * self.activation_derivate(self._net)
    gradient_w = np.transpose(dE_dNet[np.newaxis, :]) @ self._input[np.newaxis, :] 
    gradient_b = dE_dNet

    # gradient_w.resize(self.weights_matrix.shape)
    # gradient_b.resize(self.bias.shape)
    
    # calculate the dE_dO to pass to the previous layer
    current_dE_do = np.array([(np.dot(dE_dNet, self.weights_matrix[:, j])) for j in range(self.input_dim)])

    return current_dE_do, gradient_w, gradient_b



class CascadeCorrelation:
  def __init__(self, input_dim, output_dim, activation, initialization, initialization_parameters={}):
    self.input_dim = input_dim
    self.out_input_dim = None
    self.output_dim = output_dim
    self.activation = activation_functions[activation]
    self.activation_derivate = derivate_activation_functions[activation]
    self.initialization = lambda input_dim, output_dim: initialization_functions[initialization](input_dim, output_dim, **initialization_parameters)
    self.weights_matrix = None
    self.bias = None

    self._input = None
    self._net = None
    self._output = None

    self.layers = []

  def compile(self, loss=None):
    self.loss = loss

  def refresh(self):
    self.out_input_dim = sum([
      len(layer.output_dim) for layer in self.layers
    ]) + self.input_dim
    
    # init output layer weights, at compile time we have only output layer
    self.weights_matrix, self.bias = self.initialization(self.out_input_dim, self.output_dim)

  def forward_step(self, input: np.ndarray):
    internal_outputs = []

    for layer in self.layers:
      internal_outputs.append(layer.feed_forward(np.append(internal_outputs, input)))

    self._input = np.append(internal_outputs, self._input)
    self._net = np.matmul(self.weights_matrix, self._input)
    self._net = np.add(self._net, self.bias)
    self._output = self.activation(self._net)
    return self._output


  def _perceptron_delta_weights(self, input: np.array, target: np.array): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
    """
      Function that compute deltaW according to backpropagation algorithm
      Args:
          input: (np.array) initial net values
          target: the target value
      Returns:
          output: deltasW
    """

    fw_out = self.forward_step(input)
    
    dE_dO = self.loss.derivate(target, fw_out)
    dE_dNet = dE_dO * self.activation_derivate(self._net)
    gradient_w = np.transpose(dE_dNet[np.newaxis, :]) @ self._input[np.newaxis, :] 
    gradient_b = dE_dNet
    
    return (-gradient_w, -gradient_b)

  def _correlation_delta_weights(self, Vs, Es, Ds, Is): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
    """
      ...
    """
    Vmean = np.mean(Vs)
    Emean = np.mean(Es)

    # lenght of sigma_o equal to the output dim of the network
    sigma_o = np.sign(np.sum((np.array(Vs) - Vmean)*(np.array(Es) - Emean), axis=1))

    gradient_w = np.array(len(Is))
    gradient_b = 0
    
    for p, E in enumerate(Es):
      S_o = 0
      for o, s_o in enumerate(sigma_o):
        S_o += s_o*(E[o]-Emean[o])
      gradient_w += S_o*Ds[p]*Is[p]
      gradient_b += S_o*Ds[p]

    return (gradient_w, gradient_b)

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
        optimized_deltas = self.optimizer.optimize([delta], [regs])

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
        new_layer = Layer(1, "relu", "gaussian")

        new_layer_input_dim = sum([
          len(layer.output_dim) for layer in self.layers
        ]) + self.input_dim

        new_layer.initialize_weights(new_layer_input_dim)

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

            # calculate output of the new hidden unit
            internal_outputs = []
            for layer in self.layers:
              internal_outputs.append(layer.feed_forward(np.append(internal_outputs, input)))
            
            Vs.append(new_layer.forward_step(np.append(internal_outputs, input)))
            
            Is.append(np.append(internal_outputs, input))
            Ds.append(new_layer.activation_derivate(new_layer._net))

            O_p = self.forward_step(input)
            E_p = O_p - y
            Es.append(E_p)

          # at this point we have delta summed-up to all batch instances
          # let's average it
          S_p_w, S_p_b = self._correlation_delta_weights(Vs, Es, Ds, Is)     
          new_layer.weights_matrix += S_p_w
          new_layer.bias += S_p_b

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
