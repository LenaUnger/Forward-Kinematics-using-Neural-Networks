from training import n_x, n_y, n_z
from Class_Definitions import ForwardKinematics
import numpy
import math

n_x.load('x')
n_y.load('y')
n_z.load('z')

Teta1 = 0
Teta2 = 0
Teta3 = 0

# The three angles are put into a numpy array and scaled so the neural networks can work with them.
input_angles = numpy.asarray([Teta1, Teta2, Teta3])
inputs = (numpy.asfarray(input_angles) / 180) + 0.01

outputs_x = n_x.__query__(inputs)
outputs_y = n_y.__query__(inputs)
outputs_z = n_z.__query__(inputs)

label_x = (numpy.argmax(outputs_x)-200)/100
label_y = (numpy.argmax(outputs_y)-200)/100
label_z = (numpy.argmax(outputs_z)-100)/100

print("Networks answer: ", label_x, label_y, label_z)
FK = ForwardKinematics(Teta1, Teta2, Teta3)
print("Forward Kinematics answer: ", FK.__calculate__()[:3])

# Euclidean distance
euclid_distance = math.sqrt((math.pow((label_x-FK.__calculate__()[0]),2)+math.pow((label_y-FK.__calculate__()[1]),2)+math.pow((label_z-FK.__calculate__()[2]),2)))
