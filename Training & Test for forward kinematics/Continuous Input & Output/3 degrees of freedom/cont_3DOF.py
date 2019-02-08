'''
This training- and test-file trains a network to calculate the forward kinematics of a certain
standard manipulator with 3 degrees of freedom using continuous inputs and outputs. After a certain
amount of trainings, the program runs a test with 500 different angle combinations, calculating 
the mean error and the standard deviation. When training is finished, two weight matrices with the 
trained values are saved within the same path as the training file and an error function is plotted.
'''
from Class_Definitions import NeuralNetwork, ForwardKinematics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

# Create a NeuralNetwork object.
inodes = 3
hnodes = 80
onodes = 3
learnrate = 1.3
net = NeuralNetwork(inodes, hnodes, onodes, learnrate)

# Create a ForwardKinematics object.
manipulator = ForwardKinematics(0, 1, 1, 1)

# Create an array containing all values, an angle can have (integers from -180 to 180 in this case).
angles = np.linspace(-180, 180, num=361, dtype=int)

# Create empty arrays to save the mean error and deviation after each test and plot the deviations
# and mean errors in the end.
deviationlist = []
meanerrorlist = []

# Number of times, the network is trained.
epochs = 200000
# Number of trainings, after which the network is tested.
test = epochs / 10

# Train the network for epochs-number of times
for e in range(epochs):

    # For each training, 3 random angles are the input of the net. For 3 degrees of freedom, Theta 1,
    # 2 and 3 are random numbers out of the array angles.
    trainingangles = np.asarray([random.choice(angles), random.choice(angles), random.choice(angles)])
    # Set the input to a range between 0.01 and 1.00.
    traininginput = (((trainingangles + 180) / 360) * 0.99) + 0.01

    # Calculate the real result for the randomly chosen input with homogenous transformation matrices.
    result = manipulator.calculate(trainingangles[0], trainingangles[1], trainingangles[2])
    # Set the result to a range between 0.2 and 0.8 for the targets of the net.
    target = (((result[:3] + 2) / 4) * 0.6) + 0.2
    target[2] = (((result[2] + 1) / 4) * 0.6) + 0.2

    # Train the neural network.
    net.train(traininginput, target)

    # After test-number of trainings, test the network.
    if e % test == 0:

        # Create an empty array to save all errors of one training and calculate mean error and
        # standard deviation in the end.
        errorlist = []

        # Test the network for 500 angle combinations
        for i in range(500):
            # An array of 3 angles as input. Depending on the degrees of freedom, 1, 2 or 3 angles
            # are randomly chosen.
            testangles = np.asarray([random.choice(angles), random.choice(angles), random.choice(angles)])
            # Set the input to a range between 0.01 and 1.00.
            testinput = (((testangles + 180) / 360) * 0.99) + 0.01

            # Calculate the networks answer for given input.
            testoutput = net.query(testinput)
            # Resize the outputs which are numbers between 0.2 and 0.8 so they match the real answer.
            testguess = (((testoutput - 0.2) / 0.6) * 4) - 2
            testguess[2] = (((testoutput[2] - 0.2) / 0.6) * 4) - 1
            # Calculate the real result for the randomly chosen input with homogenous transformation matrices.
            testresult = manipulator.calculate(testangles[0], testangles[1], testangles[2])[:3]

            # Calculate the euclidean distance between network and function output.
            error = distance.euclidean(testguess, testresult)
            # Save the errors in an array.
            errorlist.append(error)

            # Calculate the standard deviation for the errors of the network.
            deviation = np.std(errorlist)
            # Calculate the mean error.
            meanerror = np.mean(errorlist)

            pass

        # Save the mean error and the standard deviation of the whole test run in arrays.
        deviationlist.append(deviation)
        meanerrorlist.append(meanerror)

    else:
        pass

pass

# Create an array being the size of the number of runs, the network is trained with and split
# into sections, the network is tested. This array is the x-axis of the plot showing the faults.
l = np.arange(0, epochs, epochs / 10)
# Plot the mean error for each test run.
plt.plot(l, meanerrorlist, label='meanerror')
# Plot the standard deviation for each test run.
plt.errorbar(l, meanerrorlist, yerr=deviationlist, label='standard deviation', capsize=4, ecolor='r')
# Label the axes.
plt.xlabel('number of training data')
plt.ylabel('error')
# Show the legends of the plots.
plt.legend()
# Show the created plots.
plt.show()

# Save the trained weight matrices under '3DOF_wih' and '3DOF_who'.
net.save('cont3DOF')
