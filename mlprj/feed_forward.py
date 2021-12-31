from typing import List
import numpy as np
import math
import pickle
import copy

from mlprj.utility import model_accuracy, model_loss, compiled_check

from .activations import *
from .initializers import *
from .losses import *
from .regularizers import *
from .optimizers import *

class Network:
    """
    Class of the neural network
    """
    def __init__(self, input_dim: int, layers=[]):
        """
        Args:
            input_dim: (int) the dimension of the input
            layers: (list) the layers
        """
        self.input_dim = input_dim
        self.layers = layers

        # handled by compile
        self.compiled = False
        self.loss = None
        self.regularizer = None

        self._tollerance = 1e-5
    
    def compile(self, loss=None, regularizer=None, optimizer=None):
        """
        Function that inizializes the neural network
        Args:
            loss (Loss): selected loss function
            regularizer (Regularizer): selected regularization method
            optimizer (Optimizer): optimization procedure
        """

        self.loss = loss if isinstance(loss, Loss) else loss_functions[loss]
        self.regularizer = regularizer if isinstance(regularizer, Regularizer) else regularizer_functions[regularizer]    
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else optimizer_functions[optimizer]

        prev_dim = self.input_dim 
        for layer in self.layers:
            layer.initialize_weights(prev_dim)
            prev_dim = layer.output_dim

        self.compiled = True

    @compiled_check
    def forward_step(self, net_input: np.ndarray):
        """
        A forward step of the network
        Args:
            net_input: (np.ndarray) of initial net values
        Returns:
            value: network's output
        """
        value = net_input
        for layer in self.layers:
            value = layer.forward_step(value)
        return value

    @compiled_check
    def predict(self, inputs: np.ndarray):
        """
        Perform a fast forward step over a list of input
        Args:
            inputs: (np.ndarray) array of inputs
        Returns:
            preds: network's outputs
        """
        if len(np.array(inputs).shape) == 1:
            inputs = np.expand_dims(inputs, axis=0)

        preds = []

        for net_input in inputs:
            preds.append(self.forward_step(net_input))

        return np.array(preds)

    def _bp_delta_weights(self, net_input: np.ndarray, target: np.ndarray): # return lista tuple (matrice delle derivate, vettore delle derivate biases)
        """
        Function that compute deltaW according to backpropagation algorithm
        Args:
            net_input: (np.array) initial net values
            target: (np.ndarray) the target value
        Returns:
            deltas: list of deltas (tuple: (delta_w, delta_b)) over the layers
        """
        deltas = []
        # forward phase, calcolare gli output a tutti i livelli partendo dall'input (net, out)
        fw_out = self.forward_step(net_input)
        
        dE_dO = self.loss.derivative(target, fw_out)
        # backward phase, calcolare i delta partendo dal livello di output
        for layer in reversed(self.layers):
            dE_dO, gradient_w, gradient_b = layer.backpropagate_delta(dE_dO)
            deltas.append((-gradient_w, -gradient_b))
        
        return list(reversed(deltas)) # from input to output

    def _apply_deltas(self, deltas:list):
        """
        Function that updates the weights according to W = W + deltaW
        Args:
            deltas: list of deltas
        """
        for i, (delta_w, delta_b) in enumerate(deltas):
            self.layers[i].weights_matrix += delta_w
            self.layers[i].bias += delta_b 

    def _regularize(self):
        """
        Function that returns the regularization part of the delta
        update rule w.r.t the weights of the network
        
        Returns:
            regs: (list) list of tuple of weights and biases regularization part
        """
        regs = []

        for layer in self.layers:
            reg_w = self.regularizer.regularize(layer.weights_matrix)
            reg_b = self.regularizer.regularize(layer.bias)
            regs.append((reg_w, reg_b))

        return regs

    @compiled_check
    def save_model(self, path=None):
        """
        A method to save the model in a picklable
        format and return the dictonary of the saved model

        Args:
            path: (str) if not None the path tells the location to save the model
        Returns:
            saved_model: (dict) model dictionary representation
        """
        saved_model = {}

        saved_model["layers"] = copy.deepcopy(self.layers)
        saved_model["loss"] = self.loss
        saved_model["regularizer"] = self.regularizer
        saved_model["optimizer"] = self.optimizer

        if path:
            with open(path, "wb") as f:
                pickle.dump(saved_model, f)

        return saved_model

    def load_model_from_dict(self, saved_model):
        """
        Load a saved model in a dictionary format
        """

        self.layers = saved_model["layers"]
        self.loss = saved_model["loss"]
        self.regularizer = saved_model["regularizer"]
        self.optimizer = saved_model["optimizer"]

        self.compiled = True

    def load_model_from_file(self, path):
        """
        Load a saved model in a file on the file
        system given a path name

        Args:
            path: (str) tells the location of the saved model
        """

        saved_model = {}

        with open(path, "rb") as f:
            saved_model = pickle.load(f)

        self.load_model_from_dict(saved_model)

    @compiled_check
    def training(self, training, validation=None, epochs=500, batch_size=64, early_stopping = None, verbose=False, accuracy_curve=False):
        """
        Function that performs neural network training phase
        Args:
            training: (tuple) training set, training_x, training_y
            validation: (tuple) validation set, validation_x, validation_y
            epochs: (int) number of epochs
            batch_size: (int) mini batch size
            early_stopping: (int) number of epochs before early stopping
            verbose: (boolean) if True print the epochs statistics
            accuracy_curve: (boolean) if True compute the accuracy curve of the classification model over training and validation sets
        Returns:
            history: training and validation error and accuracy for each epoch
        """

        self.optimizer.clear()

        if (early_stopping and not validation) or (early_stopping and early_stopping < 1):
            print("[Warning]: validation set not present the early stopping criteria can't be used or setted as not proper value")
            early_stopping = None

        input_tr, target_tr = training
        if validation is not None:
            input_vl, target_vl = validation
            valid_split = True
        else:
            valid_split = False
      
        if accuracy_curve:
            history = {"loss_tr" : [], "loss_vl" : [], "accuracy_tr" : [], "accuracy_vl" : []}
        else:
            history = {"loss_tr" : [], "loss_vl" : []}

        l = len(input_tr)

        if type(batch_size) is str and batch_size == "full":
            batch_size = l
        elif type(batch_size) is str and batch_size != "full":
            raise Exception()
        elif batch_size > l or batch_size <= 0:
            batch_size = l

        best_validation = np.inf
        not_improving_epochs = 0
        saved_model : dict = None
        best_epoch = 0

        for i in range(epochs):

            for batch_number in range(math.ceil(l//batch_size)):
                # sum deltas over the batch
                deltas = None

                batch_iterations = 1
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
                optimized_deltas = self.optimizer.optimize(deltas, regs, i)

                self._apply_deltas(optimized_deltas)

            epoch_error_tr = model_loss(self, self.loss, input_tr, target_tr)
            history["loss_tr"].append(epoch_error_tr)

            if accuracy_curve:
                epoch_accuracy_tr = model_accuracy(self, input_tr, target_tr)
                history["accuracy_tr"].append(epoch_accuracy_tr)

            if valid_split:
                epoch_error_vl = model_loss(self, self.loss, input_vl, target_vl)
                history["loss_vl"].append(epoch_error_vl)
                
                if accuracy_curve:
                    epoch_accuracy_tr = model_accuracy(self, input_vl, target_vl)
                    history["accuracy_vl"].append(epoch_accuracy_tr)

                if early_stopping and epoch_error_vl < best_validation and np.abs(epoch_error_vl - best_validation) > self._tollerance:
                    best_validation = epoch_error_vl
                    not_improving_epochs = 0
                    saved_model = self.save_model()
                    best_epoch = i
                else:
                    not_improving_epochs += 1            

                if verbose:
                    print(f"epoch {i}: error_tr = {epoch_error_tr} | error_vl = {epoch_error_vl}")
            else:
                if verbose:
                    print(f"epoch {i}: error_tr = {epoch_error_tr}")

            if early_stopping and not_improving_epochs >= early_stopping:
                print("early stopped !!!")
                self.load_model_from_dict(saved_model)
                history["loss_tr"] = history["loss_tr"][:best_epoch+1]
                history["loss_vl"] = history["loss_vl"][:best_epoch+1]
                if accuracy_curve:
                    history["accuracy_tr"] = history["accuracy_tr"][:best_epoch+1]
                    history["accuracy_vl"] = history["accuracy_vl"][:best_epoch+1]
                return history

        return history


class Layer:
    """
      Class of the Layer
    """
    def __init__(self, output_dim, activation, initializer):
        """
        Args:
            output_dim: (int) that indicate the dimensionality of the output
            activation: (str | ActivationFunction) name of the activation function or the class itself
            initializer: (str | Initializer) name of the initialization procedure or the class itself
        """
        self.input_dim = None
        self.output_dim = output_dim
        self.weights_matrix = None
        self.bias = None

        self.activation = activation if isinstance(activation, ActivationFunction) else activation_functions[activation]
        self.initializer = initializer if isinstance(initializer, Initializer) else initializer_functions[initializer]

        self._input = None
        self._net = None
        self._output = None

    def initialize_weights(self, input_dim):
        """
        Function that initializes the weights
        Args:
            input_dim: (int) the dimension of the layer's input
        """
        self.input_dim = input_dim
        self.weights_matrix, self.bias = self.initializer.initialize(self.input_dim, self.output_dim)

    def forward_step(self, net_input: np.ndarray):
        """
        Function that implement a layer forward step
        Args:
            net_input: (np.ndarray) initial layer's values
        Returns:
            output: layer's output
        """
        self._input = net_input
        self._net = np.matmul(self.weights_matrix, self._input)
        self._net = np.add(self._net, self.bias)
        self._output = self.activation.compute(self._net)
        return self._output

    def backpropagate_delta(self, upper_dE_dO): # return current layer
        # calculate dE_dNet for every node in the layer = dE_dO * fprime(net)
        """
        Function that backpropagate delta of this layer to the previous one
        Args:
            upper_dE_dO: (np.ndarray) derivative of E with respect to the output O for the upper layer
        Returns:
            current_dE_do: (np.ndarray) the current derivative of E with respect to O,
            gradient_w: (np.ndarray) the gradient with respect to the weights
            gradient_b:  the gradient with respect to the bias
        """
        dE_dNet = upper_dE_dO * self.activation.derivative(self._net)
        gradient_w = np.transpose(dE_dNet[np.newaxis, :]) @ self._input[np.newaxis, :] 
        gradient_b = dE_dNet
        # calculate the dE_dO to pass to the previous layer
        current_dE_do = np.array([(np.dot(dE_dNet, self.weights_matrix[:, j])) for j in range(self.input_dim)])

        return current_dE_do, gradient_w, gradient_b