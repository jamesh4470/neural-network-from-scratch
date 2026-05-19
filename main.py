import numpy

class Layer_Dense:
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
        self.output = numpy.dot(inputs, self.weights) + self.biases


# used on hidden layers
def activation_reLU(inputs):
    return numpy.maximum(0, inputs)


# used on the output layer, produces probability distribution
def activation_softmax(inputs):
    # for each row, subtract by largest value to prevent exploding values in neurons
    # shifted results are equivalent to non-shifted results
    # axis=0, operates down columns and gives result per column
    # axis=1, operates down rows and gives result per row 

    shifted = inputs - numpy.max(inputs, axis=1, keepdims=True)
    exponentiated_values = numpy.exp(shifted)
    return exponentiated_values / numpy.sum(exponentiated_values, axis=1, keepdims=True)


def loss_label(softmax_outputs, class_targets):
    softmax_outputs = numpy.clip(softmax_outputs, 1e-7, 1 - 1e-7) # prevents log(0)
    greatest_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
    loss = -numpy.log(greatest_confidences)
    average_loss = numpy.mean(loss)
    return loss


def loss_one_hot(softmax_outputs, class_targets):
    softmax_outputs = numpy.clip(softmax_outputs, 1e-7, 1 - 1e-7) # prevents log(0)
    greatest_confidences = numpy.sum(softmax_outputs * class_targets, axis=1)
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


# 2 inputs and 4 neurons
test_layer = Layer_Dense(2, 4)
print(test_layer.weights)
print(test_layer.biases)

test_softmax_output = numpy.array([[0.2, 0.2, 0.6], # class 3 greatest
                       [0.1, 0.8, 0.1], # class 2 greatest 
                       [0.5, 0.4, 0.1]]) # class 1 greatest
test_class_target = numpy.array([2, 1, 0])

print(accuracy(test_softmax_output, test_class_target))
