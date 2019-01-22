from Class_Definitions import ForwardKinematics
from Class_Definitions import NeuralNetwork
import numpy

# Initializing a neural network outputting the x-coordinate of the endeffector
inputnodes_x = 3
hiddennodes_x = 400
outputnodes_x = 401
learningrate_x = 0.3

# Initializing a neural network outputting the y-coordinate of the endeffector
inputnodes_y = 3
hiddennodes_y = 400
outputnodes_y = 401
learningrate_y = 0.3

# Initializing a neural network outputting the z-coordinate of the endeffector
inputnodes_z = 3
hiddennodes_z = 400
outputnodes_z = 401
learningrate_z = 0.3

# Creating 3 objects from the NeuralNetwork class.
# One for the x-value, one for the y-value and one for the z-value.
n_x = NeuralNetwork(inputnodes_x, hiddennodes_x, outputnodes_x, learningrate_x)
n_y = NeuralNetwork(inputnodes_y, hiddennodes_y, outputnodes_y, learningrate_y)
n_z = NeuralNetwork(inputnodes_z, hiddennodes_z, outputnodes_z, learningrate_z)

# Number of times, the three networks are trained.
epochs = 1000

for e in range(epochs):
    # The angles are drawn randomly from a discrete uniform distribution.
    TETA1 = numpy.random.randint(-180, 180)
    TETA2 = numpy.random.randint(-180, 180)
    TETA3 = numpy.random.randint(-180, 180)

    # Creating a new object from the ForwardKinematics class with the random angles for each training.
    forward_kinematics = ForwardKinematics(TETA1, TETA2, TETA3)

    # Defining the correct x, y and z-values
    x_value = forward_kinematics.__calculate__()[0].round(2)
    y_value = forward_kinematics.__calculate__()[1].round(2)
    z_value = forward_kinematics.__calculate__()[2].round(2)

    # The three angles are put into a numpy array and scaled so the neural networks can
    # work with them.
    input_angles = numpy.asarray([TETA1, TETA2, TETA3])
    inputs = (numpy.asfarray(input_angles) / 180) + 0.01

    # Setting all target values to 0.01 (0 would not work because of the sigmoid function), apart from one
    # which is set to 0.99 which is the maximum, the sigmoid function can reach.
    # This one value is the result of the calculation of forward_kinematics, which has to be scaled
    # to be an index of the output layer.
    targets_x = numpy.zeros(outputnodes_x) + 0.01
    targets_x[int(((x_value) * 100) + 200)] = 0.99
    targets_y = numpy.zeros(outputnodes_y) + 0.01
    targets_y[int(((y_value) * 100) + 200)] = 0.99
    targets_z = numpy.zeros(outputnodes_z) + 0.01
    targets_z[int(((z_value) * 100) + 100)] = 0.99

    n_x.__train__(inputs, targets_x)
    n_y.__train__(inputs, targets_y)
    n_z.__train__(inputs, targets_z)

    pass

n_x.save('x')
n_y.save('y')
n_z.save('z')

'''
TETA_1 = 0
TETA_2 = 0
TETA_3 = 0

Input_Angles = numpy.asarray([TETA_1, TETA_2, TETA_3])
Inputs = (numpy.asfarray(Input_Angles)/180) + 0.01
#n_x.load()
#n_y.load()
#n_z.load()
Outputs_x = (n_x.__query__(Inputs)).round(2)
Outputs_y = (n_y.__query__(Inputs)).round(2)
Outputs_z = (n_z.__query__(Inputs)).round(2)
label_x = (numpy.argmax(Outputs_x) - 200) / 100
label_y = (numpy.argmax(Outputs_y) - 200) / 100
label_z = (numpy.argmax(Outputs_z) - 100) / 100
print("Networks answer: ", label_x, label_y, label_z)
DH = ForwardKinematics(TETA_1, TETA_2, TETA_3)
print("Forward Kinematics answer: ", DH.__calculate__()[:3])
'''