import numpy as np
import math

def linear(x: np.ndarray):
    """ linear activation function """
    return x


def relu(x: np.ndarray):
    """ ReLU activation function """
    return np.maximum(x, 0)


def sigmoid(x: np.ndarray):
    """ Sigmoid activation function """
    ones = np.ones(x.shape)
    return np.divide(ones, np.add(ones, np.exp(-x)))


def derivate_sigmoid(x: np.ndarray):
    """ Derivative of sigmoid activation function """
    return np.multiply(sigmoid(x), np.subtract(np.ones(x.shape), sigmoid(x)))


def tanh(x: np.ndarray):
    """ Hyperbolic tangent function (TanH) """
    return np.tanh(x)


def derivative_relu(x: np.ndarray):
    """ Derivative of ReLU activation function """
    mf = lambda y: 1 if y > 0 else 0
    mf_v = np.vectorize(mf)

    return mf_v(x)


def gaussian_initialization(input_dim, output_dim):
    """Weights Gaussian initialization"""
    return np.random.normal(size = (output_dim, input_dim)), np.random.normal(size = output_dim)


def uniform_initialization(input_dim, output_dim, distribution_range):
    """Weights Uniform initialization"""
    a, b = distribution_range
    return np.random.uniform(low = a, high = b, size = (output_dim, input_dim)), np.random.uniform(low = a, high= b, size = output_dim)


def constant_initialization(input_dim, output_dim, value):
    """Weights constant initialization"""
    return np.full(shape = (output_dim, input_dim), fill_value=value), np.full(shape = output_dim, fill_value=value)


initialization_functions = {
    'gaussian': gaussian_initialization,
    'uniform': uniform_initialization,
    'constant_initialization' : constant_initialization,
}


activation_functions = {
    'linear': linear,
    'relu': relu,
    'sigmoid': sigmoid,
    #'tahn': tahn
}


derivate_activation_functions = {
    'linear': lambda _: 1,
    'relu': derivative_relu,
    'sigmoid': derivate_sigmoid,
    #'tahn': derivate_tahn
}


class Network:
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
      self.layers = layers

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
      return value
    
    def _bp_delta_weights(self, net_input: np.array, target: np.array): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
      """
        Function that compute deltaW according to backpropagation algorithm
        Args:
            net_input: (np.array) initial net values
            target: the target value
        Returns:
            output: deltasW
      """
      deltas = []
      # forward phase, calcolare gli output a tutti i livelli partendo dall'input (net, out)
      fw_out = self.forward_step(net_input)
      
      dE_dO = self.loss.derivate(target, fw_out)
      # backward phase, calcolare i delta partendo dal livello di output
      for layer in reversed(self.layers):
        dE_dO, gradient_w, gradient_b = layer.backpropagate_delta(dE_dO)
        deltas.append((-gradient_w, -gradient_b))
      
      return list(reversed(deltas)) # from input to output

    def _apply_deltas(self, deltas):
      """
        Function that updates the weights according to W = W + deltaW
        Args:
            deltas: deltaW
      """
      for i, (delta_w, delta_b) in enumerate(deltas):
        self.layers[i].weights_matrix += delta_w
        self.layers[i].bias += delta_b 


    def _regularize(self):
      """
        Function that returns the regularization part of the deltaW update rule
        Args:
            regs: (list) weights and biases regularization part
      """
      regs = []

      for layer in self.layers:
        reg_w = self.regularizer.regularize(layer.weights_matrix)
        reg_b = self.regularizer.regularize(layer.bias)
        regs.append((reg_w, reg_b))

      return regs

    def training(self, training, validation=None, epochs=500, batch_size=64):
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
              update_deltas = self._bp_delta_weights(x, y)
              for d_i, (update_delta_w, update_delta_b) in enumerate(update_deltas):
                delta_w, delta_b = deltas[d_i]
                deltas[d_i] = (delta_w + update_delta_w, delta_b + update_delta_b)

          # at this point we have delta summed-up to all batch instances
          # let's average it
          # sumgrad/mb * (mb/l) * eta
          deltas = [(delta_w/l, delta_b/l) for (delta_w, delta_b) in deltas]
          
          regs = self._regularize()

          # calculate the delta throught the optimizer
          optimized_deltas = self.optimizer.optimize(deltas, regs)

          self._apply_deltas(optimized_deltas)

        epoch_error_tr = self.compute_total_error(input_tr, target_tr)
        history["loss_tr"].append(epoch_error_tr)

        if valid_split:
          epoch_error_vl = self.compute_total_error(input_vl, target_vl)
          history["loss_vl"].append(epoch_error_vl)
          print(f"epoch {i}: error_tr = {epoch_error_tr} | error_vl = {epoch_error_vl}")
        else:
          print(f"epoch {i}: error_tr = {epoch_error_tr}")

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
      # calculate the dE_dO to pass to the previous layer
      current_dE_do = np.array([(np.dot(dE_dNet, self.weights_matrix[:, j])) for j in range(self.input_dim)])

      return current_dE_do, gradient_w, gradient_b