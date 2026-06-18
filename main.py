import numpy
from fashion_mnist.utils import mnist_reader
import matplotlib.pyplot
import time

matplotlib.use("TkAgg")

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        # weight's shape: (n_inputs, n_neurons)
        # m = number of inputs (n_inputs)
        # n = number of neurons (n_neurons)

        # weights have to be initially randomly generated and small.
        # weights cannot be zero to avoid dead nodes
        self.weights = 0.01 * numpy.random.randn(n_inputs, n_neurons)
        
        # creates horizontal vector initialized to 0
        # m = 1
        # n = n_neurons
        self.biases = numpy.zeros((1, n_neurons))

    # inputs shape: (m, n_inputs)
    # m = number of samples in the batch
    # n = n_inputs, the number of features per sample

    # biases are automatically applied to each row
    def forward(self, inputs):
        self.inputs = inputs
        self.output = numpy.dot(inputs, self.weights) + self.biases


    # outputs are neuron outputs, before reLU
    def backward(self, dLoss_dOutputs):
        self.dLoss_dWeights = numpy.dot(self.inputs.T, dLoss_dOutputs) # (n_inputs, n_neurons) 
        self.dLoss_dBiases = numpy.sum(dLoss_dOutputs, axis=0, keepdims=True) # array (1, n_neurons), one value per neuron
        self.dLoss_dInputs = numpy.dot(dLoss_dOutputs, self.weights.T) # (samples, n_inputs)


# used on hidden layers
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = numpy.maximum(0, inputs)

    # outputs are reLU outputs
    def backward(self, dLoss_dOutputs):
        # f'(x) = 1 if x > 0
        # f'(x) = 0 if x < 0
        self.dLoss_dInputs = dLoss_dOutputs.copy()
        self.dLoss_dInputs[self.inputs <= 0] = 0


class Softmax_Loss:
    # used on the output layer, produces probability distribution
    def activation_softmax(self, inputs):
        # for each row, subtract by largest value to prevent exploding values in neurons
        # shifted results are equivalent to non-shifted results
        # axis=0, operates down columns and gives result per column
        # axis=1, operates down rows and gives result per row 

        shifted = inputs - numpy.max(inputs, axis=1, keepdims=True)
        exponentiated_values = numpy.exp(shifted)
        return exponentiated_values / numpy.sum(exponentiated_values, axis=1, keepdims=True)


    # one hot
    def loss(self, softmax_outputs, class_targets):
        softmax_outputs = numpy.clip(softmax_outputs, 1e-7, 1 - 1e-7) # prevents log(0)
        true_class_probability = numpy.sum(softmax_outputs * class_targets, axis=1)
        loss = -numpy.log(true_class_probability)
        average_loss = numpy.mean(loss)
        return average_loss


    def forward(self, inputs, class_targets):
        return self.loss(self.activation_softmax(inputs), class_targets)


    # dLoss_dInputs = y_predicted - y_true
    # equivalent to subtracting 1 at the correct class position
    def backward(self, y_predicted, y_true):
        y_true = numpy.argmax(y_true, axis=1) # convert one hot to label format
        self.dLoss_dInputs = y_predicted.copy()
        self.dLoss_dInputs[range(len(y_predicted)), y_true] -= 1 # subtracts one at correct class position
        # normalize gradient
        self.dLoss_dInputs = self.dLoss_dInputs / len(y_predicted)


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, learning_rate_decay_rate=0.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate 
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.iterations = 0
        self.current_learning_rate = learning_rate


    # call before update_parameters()
    def learning_rate_decay(self):
        self.current_learning_rate = self.learning_rate * (1 / (1 + self.learning_rate_decay_rate * self.iterations))


    # momentum is the average of the signed gradients
    # cache is the average of squared gradients
    def update_parameters(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = numpy.zeros_like(layer.weights)
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_momentums= numpy.zeros_like(layer.biases)
            layer.bias_cache = numpy.zeros_like(layer.biases)

        layer.weight_momentums = (self.beta_1 * layer.weight_momentums) + ((1 - self.beta_1) * layer.dLoss_dWeights)
        layer.bias_momentums = (self.beta_1 * layer.bias_momentums) + ((1 - self.beta_1) * layer.dLoss_dBiases)

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        layer.weight_cache = (self.beta_2 * layer.weight_cache) + ((1 - self.beta_2) * layer.dLoss_dWeights**2)
        layer.bias_cache = (self.beta_2 * layer.bias_cache) + ((1 - self.beta_2) * layer.dLoss_dBiases**2)

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))

        # square root cache for r.m.s values
        layer.weights += -self.current_learning_rate * (weight_momentums_corrected / (numpy.sqrt(weight_cache_corrected) + self.epsilon))
        layer.biases += -self.current_learning_rate * (bias_momentums_corrected / (numpy.sqrt(bias_cache_corrected) + self.epsilon))


    # call after update_parameters()
    def increment_iterations(self):
        self.iterations += 1


def loss_label(softmax_outputs, class_targets):
    softmax_outputs = numpy.clip(softmax_outputs, 1e-7, 1 - 1e-7) # prevents log(0)
    greatest_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
    loss = -numpy.log(greatest_confidences)
    average_loss = numpy.mean(loss)
    return loss


# how often the largest confidence is the correct class 
def accuracy(softmax_outputs, class_targets):
    # indices of the greatest confidences
    predictions = numpy.argmax(softmax_outputs, axis=1)
    
    # handle one-hot encoded targets
    if len(class_targets.shape) == 2:
        class_targets = numpy.argmax(class_targets, axis=1)
    
    return numpy.mean(predictions == class_targets)


X_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='t10k')
y_train_onehot = numpy.eye(10)[y_train]
y_test_onehot = numpy.eye(10)[y_test]

# two hidden layers
layer_one = Dense_Layer(784, 64)
layer_two = Dense_Layer(64, 64)
layer_three = Dense_Layer(64, 10)
relu = Activation_ReLU()
softmax_loss = Softmax_Loss()
adam = Optimizer_Adam(learning_rate_decay_rate=5e-5)

for epoch in range(len(X_train)):
    x = X_train[epoch]
    y = y_train_onehot[epoch]

    layer_one_output = relu.forward(layer_one.forward(x))
    layer_two_output = relu.forward(layer_one_output)

    softmax_output = softmax_loss.activation_softmax(layer_two_output)
    loss = softmax_loss.forward(layer_two_output, y)

    print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy(softmax_output, y)}")

    softmax_loss.backward(softmax_output, y)   
