'''
This training- and test-file trains a network to calculate the forward kinematics of a certain
standard manipulator with 3 degrees of freedom using discrete inputs and outputs. After a certain
amount of trainings, the program runs a test with 500 different angle combinations, calculating the
mean error and the standard deviation. When training is finished, two weight matrices for each net
with the trained values are saved within the same path as the training file and an error function
is plotted.
'''
from Class_Definitions import NeuralNetwork, ForwardKinematics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

# Create a NeuralNetwork object outputting the x-coordinate of the endeffector.
inodes_x = 1083
hnodes_x = 200
onodes_x = 401
lr_x = 0.4
net_x = NeuralNetwork(inodes_x, hnodes_x, onodes_x, lr_x)
# Create a NeuralNetwork object outputting the y-coordinate of the endeffector.
inodes_y = 1083
hnodes_y = 200
onodes_y = 401
lr_y = 0.4
net_y = NeuralNetwork(inodes_y, hnodes_y, onodes_y, lr_y)
# Create a NeuralNetwork object outputting the z-coordinate of the endeffector.
inodes_z = 1083
hnodes_z = 200
onodes_z = 401
lr_z = 0.4
net_z = NeuralNetwork(inodes_z, hnodes_z, onodes_z, lr_z)

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

    # Depending on the degrees of freedom, 1, 2 or 3 angles are drawn randomly from the list
    # containing all integer numbers from -180 to 180.
    trainingangles = random.sample(list(angles),k=3)

    # Create the input array with value 0.01 for each element.
    traininginput = np.zeros(inodes_x) + 0.01
    # Set the value of three indices to 1.0. Those indices result from the following:
    # index1 = angle1 + 180 ; index2 = angle2 + 541 ; index3 = angle3 + 902
    traininginput[trainingangles[0] + 180]=1.0
    traininginput[trainingangles[1] + 541]=1.0
    traininginput[trainingangles[2] + 902]=1.0

    # Calculate the correct answers for the chosen angles with homogenous transformation matrices.
    trainresult = manipulator.calculate(trainingangles[0], trainingangles[1], trainingangles[2])
    x_value = trainresult[0]
    y_value = trainresult[1]
    z_value = trainresult[2]

    # Create the target array for net_x with value 0.01 for each element.
    x_target = np.zeros(onodes_x)
    # Set the value of the index which results from the correct x-value to 0.99.
    x_target[int(x_value*100 + 200)]=0.99
    # Create the target array for net_y with value 0.01 for each element.
    y_target = np.zeros(onodes_y)
    # Set the value of the index which results from the correct y-value to 0.99.
    y_target[int(y_value*100 + 200)]=0.99
    # Create the target array for net_z with value 0.01 for each element.
    z_target = np.zeros(onodes_z)
    # Set the value of the index which results from the correct z-value to 0.99.
    z_target[int(z_value*100 + 100)]=0.99

    # Train the neural networks.
    net_x.train(traininginput, x_target)
    net_y.train(traininginput, y_target)
    net_z.train(traininginput, z_target)

    # After 'test'-number of trainings, test the network.
    if e % test == 0:

        # Create an empty array to save all errors of one training and calculate mean error and
        # standard deviation in the end.
        errorlist = []

        # Test the networks for 500 angle combinations
        for i in range(500):

            # Depending on the degrees of freedom, 1, 2 or 3 angles are drawn randomly from the list
            # containing all integer numbers from -180 to 180.
            testangles = random.sample(list(angles),k=3)
            # Create the input array with value 0.01 for each element.
            testinput = np.zeros(inodes_x)+0.01
            # Set the value of three indices to 1.0. Those indices result from the following:
            # index1 = angle1 + 180 ; index2 = angle2 + 541 ; index3 = angle3 + 902
            testinput[testangles[0]+180]=1.0
            testinput[testangles[1]+541]=1.0
            testinput[testangles[2]+902]=1.0

            # Calculate the networks answers for given input.
            output_x = net_x.query(testinput)
            output_y = net_y.query(testinput)
            output_z = net_z.query(testinput)
            # The values being compared to the real result are the indices of the output array of
            # each net holding the biggest value. Resize those outputs which are numbers between
            # 0 and 400 so they match the real answer.
            x_result = (np.argmax(output_x)-200)/100
            y_result = (np.argmax(output_y) - 200) / 100
            z_result = (np.argmax(output_z) - 100) / 100
            # Calculate the real result for the randomly chosen input with homogenous transformation matrices.
            realresult = manipulator.calculate(testangles[0], testangles[1], testangles[2])

            # Calculate the euclidean distance between the result of the neural network and
            # the homogenous transformation matrix.
            error = distance.euclidean(realresult[:3], [x_result, y_result, z_result])
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
plt.plot(l, meanerrorlist, label='mean error')
# Plot the standard deviation for each test run.
plt.errorbar(l, meanerrorlist, yerr=deviationlist, label='standard deviation', capsize=4, ecolor='r')
# Label the axes.
plt.xlabel('number of training data')
plt.ylabel('error')
# Show the legends of the plots.
plt.legend()
# Show the created plots.
plt.show()

# Save the trained weight matrices under '3DOF_x_wih', '3DOF_y_wih', '3DOF_z_wih' and
# '3DOF_x_who', '3DOF_y_who' and '3DOF_z_who'.
net_x.save('3DOF_x')
net_y.save('3DOF_y')
net_z.save('3DOF_z')
