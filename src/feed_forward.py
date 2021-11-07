import numpy as np

def linear(input: np.ndarray):
    """ linear activation function """
    return input


def relu(input: np.ndarray):
    """ ReLU activation function """
    return np.maximum(input, 0)


def sigmoid(input: np.ndarray):
    """ Sigmoid activation function """
    ones = np.ones(input.shape)
    return np.divide(ones, np.add(ones, np.exp(-input)))


def derivate_sigmoid(input: np.ndarray):
    return np.multiply(sigmoid(input), np.subtract(np.ones(input.shape), sigmoid(input)))


def tanh(input: np.ndarray):
    """ Hyperbolic tangent function (TanH) """
    return np.tanh(input)


def derivative_relu(input: np.ndarray):
  mf = lambda x: 1 if x > 0 else 0
  mf_v = np.vectorize(mf)

  return mf_v(input)


def gaussian_initialization(input_dim, output_dim):
  return np.random.normal(size = (output_dim, input_dim)), np.random.normal(size = output_dim)


def uniform_initialization(input_dim, output_dim, distribution_range):
  a, b = distribution_range 
  return np.random.uniform(low = a, high = b, size = (output_dim, input_dim)), np.random.uniform(low = a, high= b, size = output_dim)


def constant_initialization(input_dim, output_dim, value): 
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
    'linear': lambda x: 1,
    'relu': derivative_relu,
    'sigmoid': derivate_sigmoid,
    #'tahn': derivate_tahn
}


class Network:
    def __init__(self, input_dim: int, layers=[]):
      """
      params:
          ...
      """
      
      self.input_dim = input_dim
      self.layers = layers

      # handled by compile
      self.loss = None
      self.regularizer = None
    
    def compile(self, loss=None, regularizer=None):
      self.loss = loss
      self.regularizer = regularizer

      prev_dim = self.input_dim 
      for layer in self.layers:
        layer.initialize_weights(prev_dim)
        prev_dim = layer.output_dim
    
    # TODO: changed forward_set in Layer
    def forward_step(self, input: np.ndarray):
      value = input
      for layer in self.layers:
        value = layer.forward_step(value)
      return value
    
    def _bp_delta_weights(self, input: np.array, target: np.array): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
      deltas = []
      # forward phase, calcolare gli output a tutti i livelli partendo dall'input (net, out)
      fw_out = self.forward_step(input)

      
      dE_dO = self.loss.derivate(target, fw_out) # -2*(target - fw_out) # last layer dE_dO modify based on the error function
      # dE_dO = -target/fw_out + (1-target)/(1-fw_out)
      # backward phase, calcolare i delta partendo dal livello di output
      dE_dO, gradient_w, gradient_b = self.layers[-1].backpropagate_delta(dE_dO)
      deltas.append((-gradient_w, -gradient_b))
      for layer in reversed(self.layers[:-1]):
        dE_dO, gradient_w, gradient_b = layer.backpropagate_delta(dE_dO)
        deltas.append((-gradient_w, -gradient_b))
      
      return list(reversed(deltas)) # from input to output

    # TODO: that function have to be done in an external function
    def _apply_delta_weights(self, deltas, learning_rate, alpha, old_deltas=None):
      if old_deltas:
        for i, (delta_w, delta_b) in enumerate(deltas):
          old_delta_w, old_delta_b = old_deltas[i]
          reg_w = self.regularizer.regularize(self.layers[i].weights_matrix)
          reg_b = self.regularizer.regularize(self.layers[i].bias)
          deltas[i] = (learning_rate*delta_w + alpha*old_delta_w + reg_w, learning_rate*delta_b + alpha*old_delta_b + reg_b)
      else:
        for i, (delta_w, delta_b) in enumerate(deltas):
          reg_w = self.regularizer.regularize(self.layers[i].weights_matrix)
          reg_b = self.regularizer.regularize(self.layers[i].bias)
          deltas[i] = (learning_rate*delta_w + reg_w, learning_rate*delta_b + reg_b)

      for i, (delta_w, delta_b) in enumerate(deltas):
        self.layers[i].weights_matrix += delta_w
        self.layers[i].bias += delta_b 

      return deltas  

    def training(self, training, validation, epochs, batch_size, learning_rate=0.1, alpha=0.1):
            
      input_tr, target_tr = training
      input_vl, target_vl = validation
      
      old_deltas = None
      history = {"loss_tr": [], "loss_vl": []}

      l = len(input_tr)

      if not batch_size:
        batch_size = l

      for i in range(epochs):

        for batch_number in range(l//batch_size):
          deltas = None

          batch_iterations = 0
          for batch_i in range(batch_number*batch_size, min(batch_number*batch_size + batch_size, l)):
            batch_iterations += 1

            x = input_tr[batch_i]
            y = target_tr[batch_i]

            if not deltas:
              deltas = self._bp_delta_weights(x, y)
            else:
              updates = self._bp_delta_weights(x, y)
              for d_i, (update_w, update_b) in enumerate(updates):
                delta_w, delta_b = deltas[d_i]
                deltas[d_i] = (delta_w + update_w, delta_b + update_b)

          # at this point we have delta summed-up to all batch instances
          # let's average it          
          deltas = [(delta_w/batch_iterations, delta_b/batch_iterations) for (delta_w, delta_b) in deltas]

          old_deltas = self._apply_delta_weights(deltas, learning_rate, alpha, old_deltas)
          
        epoch_error_tr = self.compute_total_error(input_tr, target_tr)
        epoch_error_vl = self.compute_total_error(input_vl, target_vl)

        #print(f"epoch {i}: error_tr = {epoch_error_tr} | error_vl = {epoch_error_vl}")
        history["loss_tr"].append(epoch_error_tr)
        history["loss_vl"].append(epoch_error_vl)

      return history

    def compute_total_error(self, input, target):
      total_error = 0
      for i in range(len(input)):
        total_error += self.loss.compute(target[i], self.forward_step(input[i]))
      return total_error/len(input)

class Layer:
    def __init__(self, output_dim, activation, initialization, initialization_parameters={}):
        """
        params:
            weights_matrix: a matrix of weights
            bias: a vector(or a matrix?) of biases
            activation: a string relative to the required activation function 
        """
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
      self.input_dim = input_dim
      self.weights_matrix, self.bias = self.initialization(self.input_dim, self.output_dim)

    def forward_step(self, input: np.ndarray):
        self._input = input
        self._net = np.matmul(self.weights_matrix, self._input)
        self._net = np.add(self._net, self.bias)
        self._output = self.activation(self._net)
        return self._output

    def backpropagate_delta(self, upper_dE_dO): # return current layer
      # calculate dE_dNet for every node in the layer = dE_dO * fprime(net)
      dE_dNet = upper_dE_dO * self.activation_derivate(self._net)
      gradient_w = np.transpose(dE_dNet[np.newaxis, :]) @ self._input[np.newaxis, :] 
      gradient_b = dE_dNet

      # gradient_w.resize(self.weights_matrix.shape)
      # gradient_b.resize(self.bias.shape)
      
      # calculate the dE_dO to pass to the previous layer
      current_dE_do = np.array([(np.dot(dE_dNet, self.weights_matrix[:, j])) for j in range(self.input_dim)])

      return current_dE_do, gradient_w, gradient_b