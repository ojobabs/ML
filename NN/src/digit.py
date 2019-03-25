import matplotlib.pyplot as plt
import numpy as np
import mnist_loader
import cPickle as pickle    # Add the pickle module for file saving of the W&B

# image is an Array of length 784 with values between 0 and 1.
def drawNumber(image, label):
    for i in range(len(image)):
        image[i] = image[i] * 256
        # The first column is the label
        # The rest of columns are pixels
        pixels = image
        pickle.dump(pixels, open("pixel.p", "wb"))

    # Make those columns into a array of 8-bits pixels
    # This array will be of 1D with length 784
    # The pixel intensity values are integers from 0 to 255
    pixels = np.array(pixels, dtype = 'uint8')

    # Reshape the array into 28 x 28 array (2-dimensional array)
    pixels = pixels.reshape((28, 28))

    # Plot
    plt.title('Label is {label}'.format(label = label))
    plt.imshow(pixels, cmap='gray')
    plt.show()

# Using MNIST
train_data, val_data, test_data = mnist_loader.load_data_wrapper()

# ugly patch for changing the format in which data is stored in the array
trans = zip(*test_data[0:20])

samples = trans[0]
s_list = []

for i in range(len(samples)):
    flatten = [val for sublist in samples[i] for val in sublist]
    s_list.append(flatten)

samples = s_list
y_vec = trans[1]

# example use to show the numbers in class
drawNumber(samples[5], "label 1")
drawNumber(samples[7], "label 2")
