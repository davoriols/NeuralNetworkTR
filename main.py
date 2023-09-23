import numpy as np
import mnist
import matplotlib
import matplotlib.pyplot as pyplot
from Network import Network

# parse the train images from the mnist database
trainImages = np.array(mnist.train_images())
# normalize the pixel values to pass them as activations
trainImages = (trainImages) / 255 
# reshape the array to have 600 mini sets of 100 images
trainImages = np.reshape(trainImages, (600, 100, 784))


# parse the test images from the mnist database
testImages = np.array(mnist.test_images())
# normalize the pixel values to pass them as activations
testImages = (testImages) / 255 
# reshape the array to have an array of 10000 images
testImages = np.reshape(testImages, (10000, 784))


# parse the labels form the mnist database
labels = np.array(mnist.train_labels())
# create trainLabel list with lists of zeros with the label index being 1
# for label 5, trainLabel would be [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# do this for every label
trainLabels = []
for label in labels:
    labelList = np.zeros(10)
    labelList[label] = 1
    trainLabels.append(labelList)
# reshape labels array to have 600 mini sets of 100 labels
trainLabels = np.reshape(trainLabels, (600, 100, 10))

# get the test labels
testLabels = mnist.test_labels()

# generate network object
network = Network(trainImages, trainLabels)

# iterator to know how far is the training process
iteration = 0

for epoch in range(4):
    # Wrap everything in a loop that loops through all the sets
    for minisetIndex, miniset in enumerate(trainImages):
        iteration += 1

        # first iteration of the training
        # done outside the main loop because this way, weightGradientStack and
        # biasGradientStack isn't reasigned for each iteration, I don't know how to do
        # this in another way
        network.feedForward(trainImages[0][0])
        print(f"{iteration} --> {network.cost((0, 0))}")
        print(network.layers[2].activations)

        # I create a gradientStack variable to hold the gradient values for the weights
        # and biases
        # this is done by assigning the variable to weightGradient - weightGradientStack
        # to get an array with the same shape, where each value is 0, and doesn't affect
        # the result
        weightGradient, biasGradient = network.calculateGradient((0, 0))

        weightGradientStack = weightGradient - weightGradient
        biasGradientStack = biasGradient - biasGradient

        # loops throuhg all the images in the miniset, and calculates the weightGradient
        # biasGradient, and add them to their stack
        for imageIndex, image in enumerate(miniset):
            network.feedForward(image)

            weightGradient, biasGradient = network.calculateGradient(
                (minisetIndex, imageIndex)
            )

            weightGradientStack += weightGradient
            biasGradientStack += biasGradient

        # once all images in the miniset are trained on, we get the stack, and use
        # these values for the backpropagation function

        network.backpropagation(weightGradientStack, biasGradientStack)

# export the weights and biases to their own csv, so that the network doesn't have to be
# trained every time
np.savetxt("data/weights/weights0.csv", network.layers[0].weights, delimiter=",")
np.savetxt("data/weights/weights1.csv", network.layers[1].weights, delimiter=",")
np.savetxt("data/weights/weights2.csv", network.layers[2].weights, delimiter=",")

np.savetxt("data/biases/biases0.csv", network.layers[0].biases, delimiter=",")
np.savetxt("data/biases/biases1.csv", network.layers[1].biases, delimiter=",")
np.savetxt("data/biases/biases2.csv", network.layers[2].biases, delimiter=",")


# test the network, with the test data, that has not been seen by the network before
while True:
    imageChosen = int(input("choose a test image (0-9999)"))

    # feed forward to get activations
    network.feedForward(testImages[imageChosen])
    print(network.layers[2].activations)
    activations = list(network.layers[2].activations)
    
    # represent images in matplotlib:
    # reshape the image to a (28, 28) dimension array
    image = np.reshape(testImages[imageChosen], (28, 28))
    # show the image in matplotlib
    pyplot.title(
        f"label says: {testLabels[imageChosen]} \n network says: {activations.index(max(activations))}"
    )
    pyplot.imshow(image, interpolation="nearest", cmap="gray")
    pyplot.show()
