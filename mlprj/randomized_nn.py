import numpy as np

from mlprj.utility import compiled_check, model_loss

from .activations import *
from .initializers import *
from .losses import *

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
    def __init__(self, input_dim: int, hidden_layer, output_dim, learning_bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layer = hidden_layer
        self.beta = None
        self.beta_b = None

        self.learning_bias = learning_bias

        # handled by compile
        self.compiled = False
        self.loss = None
    
    def compile(self, loss):
        """
        Function that inizializes the neural network
        Args:
            loss: selected loss function
            regularizer: selected regularization method
            optimizer: optimization procedure
        """

        self.loss = loss if isinstance(loss, Loss) else loss_functions[loss]

        prev_dim = self.input_dim 
        self.hidden_layer.initialize_weights(prev_dim)

        self.compiled = True

    @compiled_check
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

        if self.learning_bias:
            return np.matmul(self.beta, value) + self.beta_b
        else:
            return np.matmul(self.beta, value)

    @compiled_check
    def predict(self, inputs):
        preds = []

        for net_input in inputs:
            preds.append(self.forward_step(net_input))

        return np.array(preds)

    def _bias_dropout(self, H, p_d, quantile=0.5):
        mask = np.full(H.shape, 1)

        threshold = np.quantile(H, quantile)

        locs = H < threshold

        subs = np.random.binomial(1, 1-p_d, np.sum(locs))

        mask[locs] = subs

        return np.multiply(H, mask)

    @compiled_check
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
            transformed_sample = self.hidden_layer.forward_step(sample)
            transformed_train.append(transformed_sample)

        transformed_train = np.array(transformed_train)

        # Biased Dropout
        transformed_train = self._bias_dropout(transformed_train, p_d)

        if self.learning_bias:
            # add column of one
            transformed_train = np.append(transformed_train, np.full(transformed_train.shape[0], 1)[:, np.newaxis], axis=1)

        # un-regularized learner
        #output_weights = np.dot(pinv(transformed_train), target_tr)

        # regularized learner
        hidden_layer_dim = transformed_train.shape[1]
        lst_learned = np.linalg.lstsq(transformed_train.T.dot(transformed_train) + lambda_ * np.identity(hidden_layer_dim), transformed_train.T.dot(target_tr), rcond=1e-6)[0]

        if self.learning_bias:
            self.beta = lst_learned[:-1, :].T
            self.beta_b = lst_learned[-1, :]
        else:
            self.beta = lst_learned.T

        error_tr = model_loss(self, self.loss, input_tr, target_tr)

        history["loss_tr"] = error_tr

        if valid_split:
            error_vl = model_loss(self, self.loss, input_vl, target_vl)
            if verbose:
                print(f"error_tr = {error_tr} | error_vl = {error_vl}")
            history["loss_vl"] = error_vl
        else:
            if verbose:
                print(f"error_tr = {error_tr}")

        return history


class RandomizedLayer:
    """
        Class of the Layer
        Attributes:
            output_dim: (int) that indicate the dimension of the output
            activation: (str) name of the activation function
            initialization: (str) name of the initialization procedure
            initialization_parameters: (dict) hyperparameters
    """
    def __init__(self, output_dim, activation="relu"):
        self.input_dim = None # if it's not setted, not "compiled" the nn
        self.output_dim = output_dim
        self.weights_matrix = None
        self.bias = None
        self.initializer = GaussianInitializer()
        
        self.activation = activation if isinstance(activation, ActivationFunction) else activation_functions[activation]

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
        net = np.matmul(self.weights_matrix, net_input)
        net = np.add(net, self.bias)
        return self.activation.compute(net)

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
