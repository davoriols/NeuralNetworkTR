import numpy as np


class Layer:
    # generates the weights and biases of the layer
    def __init__(self, inputSize=784, outputSize=16):
        self.weights = np.random.uniform(-0.5, 0.5, (outputSize, inputSize))
        self.biases = np.random.uniform(-0.5, 0.5,(outputSize))
        self.outputSize = outputSize
        self.inputSize = inputSize

    # Sigmoid function to be used during the feed forward
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def feedLayerForward(self, input):
        self.activations = self.sigmoid(
            np.dot(self.weights, input) + self.biases)

        return self.activations

    # Cost function. only used to show how the network learns, but isn't really necessary for the learning
    def layerCost(self, labelList):
        self.costValue = 0

        # Calculate the cost by looping through each activation
        for activation in self.activations:
            self.costValue += np.power(activation - labelList[0], 2)

        return self.costValue
