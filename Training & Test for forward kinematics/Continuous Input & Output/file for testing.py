'''
With this test file, the forward kinematics of the standard manipulator are calculated for a manually given input
by the networks working with continuous input and output. If you load the weight matrices '1DOF', meaning 1 degree
of freedom, you can vary Theta3. If '2DOF' is loaded, Theta 2 and 3 can be varied and for '3DOF' all angles are
variable. The only lines you need to change to test the forward kinematics with certain angles and degrees of
freedom are lines 12, 13, 14 and 23.
'''
from Class_Definitions import NeuralNetwork, ForwardKinematics
import numpy as np

# Type in the angles, for which the forward kinematics should be solved, in degree.
Theta1 = 0
Theta2 = 0
Theta3 = 23

# Initialize a neural network, named testnet, with the same parameters as the one you want to test.
inputnodes = 3
hiddennodes = 80
outputnodes = 3
learningrate = 1.3
testnet = NeuralNetwork(inputnodes, hiddennodes, outputnodes, learningrate)
# Load the weight matrices from the net you want to test into the testnet.
testnet.load('1DOF')

# Simulate a standard manipulator by initializing a ForwardKinematics object. It should have the same parameters
# like the one, the net was trained with.
manipulator = ForwardKinematics(0, 1, 1, 1)

# Set the input to a range between 0.01 and 1.00.
inputs = (((np.asarray([Theta1, Theta2, Theta3]) + 180) / 360) * 0.99) + 0.01
# Calculate the networks answer for given input.
output = testnet.query(inputs)
# Resize the outputs which are numbers between 0.2 and 0.8 so they match the real answer.
netanswer = (((output - 0.2) / 0.6) * 4) - 2
netanswer[2] = (((output[2] - 0.2) / 0.6) * 4) - 1
# Calculate the solutions of the forward kinematics for given input with homogenous transformation matrices.
realanswer = manipulator.calculate(Theta1, Theta2, Theta3)[:3]

print("Networks answer:   ", "        x:", '%.3f' % float(netanswer[0]), "  y:", '%.3f' % float(netanswer[1]), " z:",
      '%.3f' % float(netanswer[2]))
print("Forward Kinematics answer: ", "x:", '%.3f' % realanswer[0], "  y:", '%.3f' % realanswer[1], " z:",
      '%.3f' % realanswer[2])
