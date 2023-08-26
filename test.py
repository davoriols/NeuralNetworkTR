import numpy as np
from numpy import genfromtxt
import mnist
from Network import Network
import matplotlib.pyplot as pyplot

# Configurationt options: 

# change to True to used saved training data, or False to data from previous training
useTrainData = False

# change to True to show a pyplot of the incorrect gueses, of False to not show
showResults = False

# parse the test images from the mnist database
testImages = np.array(mnist.test_images())
# normalize the pixel values to pass them as activations
testImages = (testImages - np.min(testImages)) / (
    np.max(testImages) - np.min(testImages)
)
# reshape the array to have 600 mini sets of 100 images
testImages = np.reshape(testImages, (10000, 784))

# import test labels
testLabels = mnist.test_labels()

# generate network object
# for the methods we use to test, we don't accually need the trainData, but the network
# requires it, so we just pass None
network = Network(None, None)

# checks if what training data we want to use
if useTrainData:
    # get the pretrained weights and biases
    network.layers[0].weights = genfromtxt(
        "trainedData/weights/weights0.csv", delimiter=",", dtype=None
    )

    network.layers[1].weights = genfromtxt(
        "trainedData/weights/weights1.csv", delimiter=",", dtype=None
    )

    network.layers[2].weights = genfromtxt(
        "trainedData/weights/weights2.csv", delimiter=",", dtype=None
    )

    # do the same for weights
    network.layers[0].biases = genfromtxt(
        "trainedData/biases/biases0.csv", delimiter=",", dtype=None
    )

    network.layers[1].biases = genfromtxt(
        "trainedData/biases/biases1.csv", delimiter=",", dtype=None
    )

    network.layers[2].biases = genfromtxt(
        "trainedData/biases/biases2.csv", delimiter=",", dtype=None
    )

else:
    # get the pretrained weights and biases
    network.layers[0].weights = genfromtxt(
        "data/weights/weights0.csv", delimiter=",", dtype=None
    )

    network.layers[1].weights = genfromtxt(
        "data/weights/weights1.csv", delimiter=",", dtype=None
    )

    network.layers[2].weights = genfromtxt(
        "data/weights/weights2.csv", delimiter=",", dtype=None
    )

    # do the same for weights
    network.layers[0].biases = genfromtxt(
        "data/biases/biases0.csv", delimiter=",", dtype=None
    )

    network.layers[1].biases = genfromtxt(
        "data/biases/biases1.csv", delimiter=",", dtype=None
    )

    network.layers[2].biases = genfromtxt(
        "data/biases/biases2.csv", delimiter=",", dtype=None
    )

network.feedForward(testImages[600])
print(network.layers[2].activations)
print(testLabels[600])

labels = list(network.layers[2].activations)

print(labels.index(max(labels)))


totalGuesses = 0
correctGuesses = 0


for imageIndex, image in enumerate(testImages):
    totalGuesses += 1

    network.feedForward(image)

    activations = list(network.layers[2].activations)

    if activations.index(max(activations)) == testLabels[imageIndex]:
        correctGuesses += 1

    else:
        if showResults:
            image = np.reshape(image, (28, 28))
            # show the image in matplotlib
            pyplot.title(
                f"survey says: {testLabels[imageIndex]} \n network says: {activations.index(max(activations))}"
           )
            # pyplot.title(f"network says: {activations.index(max(activations))}")
            pyplot.imshow(image, interpolation="nearest", cmap="gray")
            pyplot.show()

    print(f"correct guesses: {correctGuesses}")
    print(f"total guessses:  {totalGuesses}")
    print(f"accuracy:        {correctGuesses / totalGuesses * 100}")
