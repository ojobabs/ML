# Project: Using Neural Networks to Recognize Handwritten Digits
# Date: March 8th, 2019
# Original Author: Michael Nielsen
# Modified By: Oscar Alberto Carreno Gutierrez

# This script implements a gradient descent learning algorithm for
# a feedforward Neural Network with gradients calculated using
# backpropagation and the mean square cost function.

import numpy as np          # Add the numpy module as an alias (np) pointing to numpy
import mnist_loader         # Add the loader script to use the MNIST dataset in the NN
import cPickle as pickle    # Add the pickle module for file saving of the W&B
import random               # Add the random module to initialize the biases & weights
import sys                  # Add the sys module for all methods and functions redirection
import os                   # Add the os module interface with ubuntu
import scipy
from scipy import ndimage
from time import sleep

################################################################################

class Network(object):      # Blueprint for our NN with all the properties and functions

    def __init__(self, sizes, learn, save):     # Constructor of our NN, initializes our variables
        self.learn = learn                      # Decides to learn from random or dumped files
        self.save = save                        # Decides to save the final values for saving
        self.num_layers = len(sizes)            # Return the number of layers in sizes
        self.sizes = sizes                      # Instances declared of the initial settings

        if learn == True:
            # Utilizes the Random library to generate a float with a Mean of 0 and a Std Dev of 1 using a
            # Gaussian distribution and uses the zip() function to pair elements of the w_jk layers.
            self.biases = [np.random.randn(y, 1)    #
                          for y in sizes[1:]]       #
            self.weights = [np.random.randn(y, x)   #
                          for x, y in zip(sizes[:-1], sizes[1:])]
        elif learn == False:
            # Utilizes the cPickle library to import the Biases and Weights from previous learning iterations
            # as initial values insted of randomizing them.
            self.biases = pickle.load( open("msc_biases.p", "rb"))
            self.weights = pickle.load( open("msc_weights.p", "rb"))

    def SGD(self, training_data, epochs, mini_batch_size, learn_rate, test_data = None):
        # If a test data is selected save the quantity of the test dataset. Save the quantity of values of
        # the training dataset. For the amount of epochs create a random shuffle of the training data into
        # a mini batch of the size selected, then for each mini_batch apply gradient descent and if a test
        # data is selected make an evaluation of the data with the updated weights. If the save function is
        # enabled the weights and biases at the end are dumped into a .p file for further use.

        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learn_rate)
            if test_data:
                print "  Epoch {0}: {1} / {2}".format(j+1, self.evaluate(test_data), n_test)
            else:
                print "  Epoch {0} complete".format(j)

        print "\n"
        save_initval(self.weights, self.biases)

    def feedforward(self, a):
        # Uses the paired values of the B&W of the layer and calculates the sigmoid for each neuron
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def update_mini_batch(self, mini_batch, learn_rate):
        # Creates two numpy arrays (nabla_b. nabla_w) and fills them with 0's, then it creates a delta
        # for both and runs the backpropagation function for both arrays, after that it sums the original
        # arrays with the backpropagated ones for the mini_batch. after the loop it updates the new values
        # to the self biases and weights (new_weight = (weight-(learning_rate/size of batch))*nabla_weight).

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (learn_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (learn_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        # Creates two numpy arrays (nabla_b. nabla_w) and fills them with 0's
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Creates the activation layer, the lists of the activations to be done, as well as the z vector list
        activation = x
        activations = [x]
        zs = []

        # "Feedforward" Calculates the z vector by adding the dot product of the weights and the activation
        # layer and appends the z vector to the lists that stores them, then it obtains the sigmoid of z
        # and appends the value of this activation into the list of activations.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Calculate the Output Error by computing the vector delta which is equal to the multiplication of
        # The Vector of cost derivatives of the activations and the derivative of the sigmoid function of z
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Backpropagate the error by computing, for each layer, the delta which is equal to the dot product
        # of the result of the transpose of the weights by the previous delta and the sigmoid_prime result
        # of z, finally calculate the Output of the Weight by multiplying the activation layer with the delta
        # with the bias being the delta by itself.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        # Returns a vector of partial derivatives for the output activations of two numpy arrays
        return (output_activations - y)

    def evaluate(self, test_data):
        # It is called by the SGD function to evaluate the current network outputs (the highest activation
        # output neuron is considered the result) to the test data and get the total of correct answers
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def eval_digit(self):
        ex_digit = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("3.png", flatten = True)))
        test = self.feedforward(ex_digit)

        print("  Prediction: " + str(np.unravel_index(np.argmax(test), test.shape)))


################################################################################

# Calculate the Sigmoid Function and its Prime and return its value
def sigmoid(z):
    return (1.0) / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Checks with the OS to see if the previously dumpedfiles exist
def prev_initval():
    global learn
    if learn == False:                             # Only does this if you want to use previous values
        w_exist = os.path.isfile('msc_weights.p')  # Check if the Weights for the MSCF exist
        b_exist = os.path.isfile('msc_biases.p')   # Check if the Biases for the MSCF exist
        if w_exist == False or b_exist == False:   # If any of the files is missing start again learning
            learn = True
            print "  Cannot charge previous values, file/s missing, starting with saved values \n"
            print "  " + "*" * 78 + "\n"
        else:
            print "  Files succesfully found, learning with saved initial values \n"
            print "  " + "*" * 78 + "\n"
    else:
        print "  Learning will be made with random initial values for the weights and biases. \n"
        print "  " + "*" * 78 + "\n"

# Uses the cPickle format to save the final values to use them in further NN as initial values
def save_initval(weights, biases):
    if save == True:
        pickle.dump(weights, open("msc_weights.p", "wb"))   # Save the values of the weights after to a file
        pickle.dump(biases, open("msc_biases.p", "wb"))     # Save the final valuues of the biases to a file

# Main Program execution for the NN
def main():
    # Verifies that previous values exist in case that learn == False (to use dumped values)
    prev_initval()
    # Loads the MNIST dataset to the script
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # Defines the object with the values defined in the declaration of variables
    nn = Network([input_layer, hidden_l1, output_layer], learn, save)
    # Executes the Stochastic Gradient Descent Neural Network with the valures defined.
    nn.SGD(training_data, epochs, mini_batch, learn_rate, test_data = test_data)

    # nn.eval_digit()


class FlushFile(object):
    # Write-only flushing wrapper for file-type objects.
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


if __name__ == "__main__":
    # Declaration of variables for the Neural Network
    learn        = True  # Toggle to either learn with random values (True) or use the dumped values (False)
    save         = False # Toggle to either save the new w&b or leave the previous ones if existing
    input_layer  = 784   # No. of neurons in input layer (28x28px = 784 for MNIST images)
    hidden_l1    = 30    # No. of neurons in the first hidden layer
    output_layer = 10    # No. of neurons in the output layer (Numbers: 0 - 9 = 10 possibilities)
    epochs       = 10     # No. of epochs to run the Neural Network
    mini_batch   = 10    # Size of mini batches for random training
    learn_rate   = 3.0   # Learning Rate for the NN, bigger learning rate may cause divergence

    print("\n Running the Neural Network with: \n"
          "     > Epochs: " + str(epochs) + "\n" +
          "     > A Mini Batch with a size of: " + str(mini_batch) + "\n" +
          "     > A Learning Rate of: " + str(learn_rate) + "\n")

    if "-l" in sys.argv: # If selected, the NN will learn from previous dumped values
        learn    = False
        print "  Learning will be made with previously saved initial values. \n"
        print "  " + "*" * 78 + "\n"

    if "-s" in sys.argv: # If selected, it will save the last W&B found in the NN
        save     = True
        print "  Final Weights and Biases will be saved in the root folder. (msc_biases.p & msc_weights.p) \n"
        print "  " + "*" * 78 + "\n"

    # Replaces stdout with an automatically flushing version
    sys.stdout = FlushFile(sys.__stdout__)
    sys.stderr = FlushFile(sys.__stderr__)

    main()              # Runs the NN with the parameters declared in the initialization
