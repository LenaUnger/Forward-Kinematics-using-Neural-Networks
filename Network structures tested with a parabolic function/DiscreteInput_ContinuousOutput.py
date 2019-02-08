from Class_Definitions import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import random

# Create a NeuralNetwork object.
inodes = 101
hnodes = 60
onodes = 1
lr = 0.4
net = NeuralNetwork(inodes, hnodes, onodes, lr)

# Create empty arrays to plot the deviations and mean errors over the whole training
# process when finished.
deviationlist = []
meanerrorlist = []

# Number of times, the network is trained.
epochs = 10000
# Number of trainings, after which the network is tested.
test = epochs / 5

for e in range(epochs+1):

    # Chose an input value of the range from 0 to 1, discretised to 2 decimals.
    x = random.choice(np.linspace(0, 1, num=101))
    # Create the input array with value 0.01 for each element
    xlist = np.zeros(inodes)+0.01
    # Set the value of the index which results from the multiplication of x with 100 to 1.0.
    xlist[int(x * 100)] = 1.00

    # Define the target for the network avoiding 0 because it can not be reached by the
    # sigmoid activation function.
    y = (3 * x * (1 - x))
    target = y + 0.1

    # Train the neural network.
    net.train(xlist, target)

    # After 'test'-number of trainings, test the network.
    if e % test == 0:

        # Create empty arrays for each test to plot the function calculated by the network
        # and calculate mean error and deviation.
        outputarray = []
        errorlist = []

        # Test the network for 300 input values between 0 and 1.
        input = np.linspace(0, 1, num=300)
        for i in range(len(input)):
            # Create the input array the same way as in training.
            inputlist = np.zeros(inodes)+0.01
            inputlist[int(input[i] * 100)] = 1.00
            # Subtract 0.1 from the output, to compensate the previously set target.
            output = float(net.query(inputlist))-0.1
            # Save the outputs in an array.
            outputarray.append(output)
            # Calculate the absolute difference between network and function output.
            error = abs(output - (3 * input[i] * (1 - input[i])))
            # Save the errors in an array.
            errorlist.append(error)
            # Calculate the standard deviation for the errors of the network.
            d = np.std(errorlist)
            # Calculate the mean error.
            m = np.mean(errorlist)
            pass

        # Save the mean error and the standard deviation of the whole test run in arrays.
        deviationlist.append(d)
        meanerrorlist.append(m)

        # Plot the function for each test run.
        plt.plot(input, outputarray, label=e)

    else:
        pass

    pass

# Plot the real function.
plt.plot(input, 3 * input * (1 - input), label='y = 3x*(1-x)', color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
# Show all previously created plots in one window.
plt.show()

# Create an array being the size of the number of runs, the network is trained with and split
# into sections, the network is tested. This array is the x-axis of the plot showing the faults.
l = np.arange(0, epochs+1, epochs / 5)
# Plot the mean error for each test run.
plt.plot(l, meanerrorlist, label='mean error')
# Plot the standard deviation for each test run.
plt.errorbar(l, meanerrorlist, yerr=deviationlist, label='standard deviation', capsize=4, ecolor='r')
plt.xlabel('number of training data')
plt.ylabel('error')
plt.legend()
# Show the mean errors and standard deviations in one window.
plt.show()

# Save the trained weight matrices under 'disc_cont_wih' and 'disc_cont_who'.
net.save('disc_cont')
