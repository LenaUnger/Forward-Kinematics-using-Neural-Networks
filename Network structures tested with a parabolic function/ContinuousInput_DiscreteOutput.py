from Class_Definitions import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import random

# Create a NeuralNetwork object.
inodes = 1
hnodes = 40
onodes = 76
lr = 0.4
net = NeuralNetwork(inodes, hnodes, onodes, lr)

# Create empty arrays to plot the deviations and mean errors over the whole training
# process when finished.
deviationlist = []
meanerrorlist = []

# Number of times, the network is trained.
epochs = 40000
# Number of trainings, after which the network is tested.
test = epochs / 5

for e in range(epochs+1):

    # Set the input to a range between 0.01 and 1.00.
    x = ((random.random()) * 0.99) + 0.01

    # Calculate the result for the given input with the function. Round the result to 2 decimals
    # which is the accuracy, the network is limited to due to the discretization of the output.
    y = (3 * x * (1 - x)).__round__(2)
    # Create the target array with value 0.01.
    targetlist = np.zeros(onodes) + 0.01
    # Set the value of the index which results from the multiplication of y with 100 to 0.99.
    targetlist[int(y * 100)] = 0.99

    # Train the neural network.
    net.train(x, targetlist)

    # After 'test'-number of trainings, test the network.
    if e % test == 0:

        # Create empty arrays for each test to plot the function calculated by the network
        # and calculate mean error and deviation.
        outputarray = []
        errorlist = []

        # Test the network for 300 input values between 0 and 1.
        input = (np.linspace(0, 1, num=300))
        for i in range(len(input)):
            # Ask the network for the result, being an array similar to the target array.
            result = net.query(input[i])
            # The value being compared to the result of the function is the index of the result
            # array of the network holding the biggest value divided by 100.
            output = (np.argmax(result)) / 100
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
plt.legend()
# Show the mean errors and standard deviations in one window.
plt.show()

# Save the trained weight matrices under 'cont_disc_wih' and 'cont_disc_who'.
net.save('cont_disc')
