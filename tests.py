from main import *
import numpy

def test_accuracy():
    # 2 inputs and 4 neurons
    test_layer = Dense_Layer(2, 4)
    print(test_layer.weights)
    print(test_layer.biases)

    test_softmax_output = numpy.array([[0.2, 0.2, 0.6], # class 3 greatest
                           [0.1, 0.8, 0.1], # class 2 greatest 
                           [0.5, 0.4, 0.1]]) # class 1 greatest
    test_class_target = numpy.array([2, 1, 0])

    print(accuracy(test_softmax_output, test_class_target))


def visualize_data():
    for i in range(10):
        print(y_train[i])
        print(y_train_onehot[i])
        img_data = x_train[i].reshape(28, 28)
        matplotlib.pyplot.imshow(img_data)
        matplotlib.pyplot.show()

visualize_data()
