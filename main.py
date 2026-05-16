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

def reLU(inputs):
    return numpy.maximum(0, inputs)

# 2 inputs and 4 neurons
test_layer = Layer_Dense(2, 4)
print(test_layer.weights)
print(test_layer.biases)
