import numpy as np
from Layer import Layer


class Network:
    def __init__(self, trainInput, trainLabels, learningRate=-0.01):
        # network with four layers of which the first layer is just the input
        # therefore we don't declare it since it doesn't have weights or biases
        self.layers = [Layer(784, 20), Layer(20, 20), Layer(20, 10)]
        self.trainInput = trainInput
        self.trainLabels = trainLabels
        self.learningRate = learningRate

    # Function to feedForward the network
    # Uses the output of feedLayerForward() as the input for the next layer
    def feedForward(self, input):
        self.layers[2].feedLayerForward(
            self.layers[1].feedLayerForward(self.layers[0].feedLayerForward(input))
        )

        return self.layers[2].activations

    # function to calculate the cost of the network, not necessary for training, but
    # used to measure it
    def cost(self, image):
        self.costValue = 0

        # Calculate the cost by looping through each activation
        for activationIndex, activation in enumerate(self.layers[-1].activations):
            self.costValue += np.power(
                activation - self.trainLabels[image[0]][image[1]][activationIndex], 2
            )

        return self.costValue

    # function to calculate the deltas of the network ahead of time to be used in the
    # calculateGradient function
    def calculateDeltas(self, image):
        # empty list of lists, to which the deltas will be appended to the list of its
        # corresponding layer
        # \delta^2_3 will be in deltas[2][3]
        deltas = [[], [], []]

        for layerIndex, layer in zip([2, 1, 0], self.layers):
            # calculates the deltas of the last layer as they are calculated differently
            if layerIndex == 2:
                for activationIndex, activation in enumerate(
                    self.layers[-1].activations
                ):
                    deltas[2].append(
                        activation
                        * (1 - activation)
                        * 2
                        * (
                            activation
                            - self.trainLabels[image[0]][image[1]][activationIndex]
                        )
                    )

            # calculates the deltas for the rest of the layers
            else:
                for activationIndex, activation in enumerate(
                    self.layers[layerIndex].activations
                ):
                    delta = 0
                    for neuronIndex, neuron in enumerate(
                        self.layers[layerIndex + 1].activations
                    ):
                        delta += (
                            deltas[layerIndex + 1][neuronIndex]
                            * self.layers[layerIndex + 1].weights[neuronIndex][
                                activationIndex
                            ]
                            * activation
                            * (1 - activation)
                        )
                    deltas[layerIndex].append(delta)
        return deltas

    # function to calculate the gradient vector for all the weights and biases
    # return a weightGradient and a biasGradient list containing the a multiple
    # of how much these values have to be changed in the backpropagation function
    #
    # therefore, the gradient vector is divided into two, to facilitate looping through
    # weights and biases separately
    def calculateGradient(self, image):
        # generate an empty array whose values will be changed to the
        # gradient value of the weight
        weightGradient = [[], [], []]

        weightGradient[1] = np.zeros(
            (self.layers[1].outputSize, self.layers[1].inputSize), dtype=object
        )

        weightGradient[2] = np.zeros(
            (self.layers[2].outputSize, self.layers[2].inputSize), dtype=object
        )

        biasGradient = [[], [], []]

        biasGradient[0] = np.zeros((self.layers[0].outputSize), dtype=object)

        biasGradient[1] = np.zeros((self.layers[1].outputSize), dtype=object)

        biasGradient[2] = np.zeros((self.layers[2].outputSize), dtype=object)

        # calculate deltas with calculateDeltas function:
        deltas = self.calculateDeltas(image)

        # calculate the amount we have to change the weights:
        for layerIndex, layer in enumerate(self.layers):
            # checks if it is layer 0 because it need the input as an activation
            if layerIndex == 0:
                # we calculate the gradient value for a weight w_{k,j} following
                # a^{L-1}_j * \delta^L_k
                # we use trainInput as the activation from the previous layer as
                # this is not stored as an activation

                # we reshape the information to be able to do the dot product, saving a
                # loop

                imageShaped = self.trainInput[image[0]][image[1]].reshape(1, 784)
                deltasShaped = np.array(deltas[0]).reshape(
                    np.array(deltas[0]).shape[0], -1
                )
                weightGradient[0] = self.learningRate * np.dot(
                    deltasShaped, imageShaped
                )

            else:
                # create variables that store the information in the correct shape to do
                # the dot product
                activationsShaped = self.layers[layerIndex - 1].activations
                activationsShaped = activationsShaped.reshape(
                    1, activationsShaped.shape[0]
                )
                deltasShaped = np.array(deltas[layerIndex]).reshape(
                    np.array(deltas[layerIndex]).shape[0], -1
                )

                # we do the dot product to calculate the gradient
                weightGradient[layerIndex] = self.learningRate * np.dot(
                    deltasShaped, activationsShaped
                )

            # we do the same for the biases, but since there isn't any operations to do,
            # we can simply asign the deltas

            biasGradient[layerIndex] = self.learningRate * np.array(deltas[layerIndex])


        return np.array(weightGradient, dtype=object), np.array(
            biasGradient, dtype=object
        )

    # function to modify the weights and biases by a multiple of the gradient
    # this gradient is en theory a average of all the gradients of the mini set
    def backpropagation(self, weightGradient, biasGradient):
        # loop through the weights and subtract a multiple of the gradient value for
        # each weight
        for layerIndex, layer in enumerate(self.layers):
            for rowIndex, row in enumerate(layer.weights):
                for weightIndex, weight in enumerate(row):
                    self.layers[layerIndex].weights[rowIndex][
                        weightIndex
                    ] += weightGradient[layerIndex][rowIndex][weightIndex]

            # loop through the weights and subtract a multiple of the gradient value for
            # each bias
            for biasIndex, bias in enumerate(layer.biases):
                self.layers[layerIndex].biases[biasIndex] += biasGradient[layerIndex][
                    biasIndex
                ]
