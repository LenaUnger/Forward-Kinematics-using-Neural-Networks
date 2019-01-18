# Import numpy package (numerical Python) for matrix operations
import numpy
# Import math package for trigonometrical functions
import math
# Import scipy.special package for the sigmoid function
import scipy.special

# Set the way, numpy objects (like matrices, arrays) are displayed to floats with 2 decimals
numpy.set_printoptions(precision=2, formatter={'float_kind': '{:0.2f}'.format})


class ForwardKinematics:
    """
    This class contains the calculation of the position of the endeffector with matrix multiplication.
    This class is designed for an articulated manipulator with three revolute joints and four links.
    The length of link 1 to 3 is set to 1, link 0 has a length of 0. The angles of the revolute
    joints are variables, defined for each object in the __init__- method. The transformation matrices
    follow the Denavit Hartenberg transformation for describing robot kinematics.
    To change the type or orientation of the joints or the link lengths, one needs to alter the
    transformation matrices tn_n-1. To add joints and links, new transformation matrices tn_n-1 need to
    be defined and added to the multiplication of the matrix tn_0.
    """

    def __init__(self, angle1, angle2, angle3):
        """
        To initialize an object of this class, the three angles defining the orientation of the
        revolute joints need to be defined. All three parameters need to be integers in the range
        [-180, 180].
        """
        """
        angle1: angle1 is the angle of the first revolute joint.
        angle1: angle2 is the angle of the second revolute joint.
        angle1: angle3 is the angle of the third revolute joint.
        """
        self.angle_1 = angle1
        self.angle_2 = angle2
        self.angle_3 = angle3

        # As the mathematical operations of the math package work with angles of the radiant type,
        # the given degree angles need to be transformed into radiant.
        self.angle_1 = numpy.deg2rad(self.angle_1)
        self.angle_2 = numpy.deg2rad(self.angle_2)
        self.angle_3 = numpy.deg2rad(self.angle_3)


        pass

    def __calculate__(self):
        """
        This method contains the calculation of the transformation matrices and the
        position of the endeffector. At first, the transformation matrices from one joint to
        another are calculated, then the multiplication of all matrices to get the transformation
        matrix from the basis to the endeffector and finally, by a multiplication of the whole
        transformation matrix with the origin of the coordinate system of the endeffector (the
        tool center point), the position of the endeffector is described by coordinates of
        the base coordinate system.
        """

        # transformation matrices from link n to link n-1
        t1_0 = numpy.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
        t2_1 = numpy.array([[math.cos(self.angle_1), 0, math.sin(self.angle_1), 0],
                            [math.sin(self.angle_1), 0, -(math.cos(self.angle_1)), 0],
                            [0, 1, 0, 1],
                            [0, 0, 0, 1]], dtype=float)
        t3_2 = numpy.array([[-(math.sin(self.angle_2)), -(math.cos(self.angle_2)), 0, -(math.sin(self.angle_2))],
                            [math.cos(self.angle_2), -(math.sin(self.angle_2)), 0, math.cos(self.angle_2)],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
        t4_3 = numpy.array([[math.cos(self.angle_3), -(math.sin(self.angle_3)), 0, math.cos(self.angle_3)],
                            [math.sin(self.angle_3), math.cos(self.angle_3), 0, math.sin(self.angle_3)],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)

        # Calculation of the transformation matrix from link 4 to 0
        t4_0 = t1_0.dot(t2_1).dot(t3_2).dot(t4_3)

        # endpoint is the origin of the coordinate system of the endeffector which has to be
        # described by coordinates of the base coordinate system.
        endpoint = numpy.array([0, 0, 0, 1])

        # Calculation of the position of the endeffector in coordinates of the base system.
        endposition = numpy.dot(t4_0, endpoint)

        # Returning the 4-dimensional vector
        return endposition

        pass

class NeuralNetwork:
    """
    This class contains the neural network. The number of layers is set to three - one input layer,
    one hidden layer and one output layer. The number of neurons per layer as well as the learning rate
    is variable in this class, set to a certain value for each object.
    """

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """
        To initialize an object of this class, the number of input, hidden and output neurons as
        well as the learning rate have to be indicated. Apart from the number of layers, the size
        regarding the neurons of the neural network is variable.
        """
        """
        inputnodes: Sets number of input neurons.
        hiddennodes: Sets number of hidden neurons.
        outputnodes: Sets number of output neurons.
        learningrate: Sets the value of the learning rate of the neural network. The learning rate
        has to exceed 0 and fall below 1. A smaller learning rate means the steps the neural network
        takes when training with the gradient descend are smaller, bigger vice versa. 
        """

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # The initialized weights are drawn from a normal distribution with the standard deviation being
        # the square root of the number of neurons from the hidden layer (self.wih) or the output layer
        # (self.who)[Rashid, p. 92]. Regarding the size of the matrix: There are as many rows as there
        # are neurons in the layer to which the weights are connected ans as many columns as there are
        # neurons in the layer from where the weights come.
        # self.wih is the weight matrix between the input and the hidden layer.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # self.who is the weight matrix between the hidden and the output layer.
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Activation function describes the activation of the neurons. Scipy.special.expit(x) is a sigmoid
        # function, also known as the logistic function, defined as expit(x) = 1/(1+exp(-x)).
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def __train__(self, inputs_list, targets_list):
        """
        This method contains the training of the neural network. It includes 2 parts:
        First, the inputs are passed through the neural network, being multiplied with the weights
        for the input of each layer and calculated with the sigmoid function for the output
        of each layer. Then, the errors of the hidden and the output layer using the backpropagation
        algorithm and the values that need to be added to the existing weights are calculated.
        """
        """
        inputs_list: The list of data going into the neural network.
        targets_list: The list of data giving the neural network its target values for calculating the 
        output error. 
        """

        # Converting the input and target lists into arrays with one column and as many
        # rows as there are values, ensuring the functionality of the matrix multiplication.
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # The inputs of the hidden layer are the combined and moderated signals, so a matrix
        # multiplication of the weight matrix with the inputs.
        hidden_inputs = numpy.dot(self.wih, inputs)
        # The hidden outputs are the sigmoid function of the hidden inputs.
        hidden_outputs = self.activation_function(hidden_inputs)

        # The inputs of the output layer are the combined and moderated signals, so a matrix
        # multiplication of the weight matrix with the hidden outputs.
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # The final outputs are the sigmoid function of the final inputs.
        final_outputs = self.activation_function(final_inputs)

        # Calculation of the output errors.
        output_errors = targets - final_outputs
        # Calculation of the hidden errors splitting up the output errors according to the
        # weights from the hidden to the output layer.
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Updating the weight matrices with the formula coming from the gradient descend method.
        # This formula is derived in the chapter 'the backpropagation algorithm' of the thesis.
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        pass

    def save(self):
        '''
        This method stores the trained weight matrices in the file 'saved_wih.npy' (self.wih)
        or 'saved_who.npy' (self.who).
        '''
        numpy.save('saved_wih.npy', self.wih)
        numpy.save('saved_who.npy', self.who)
        pass

    def load(self):
        '''
        This method loads the previously saved weight matrices to either train them further
        or let the trained neural network solve a problem.
        '''
        self.wih = numpy.load('saved_wih.npy')
        self.who = numpy.load('saved_who.npy')
        pass

    def __query__(self, inputs_list):
        '''
        This method queries the answer of the neural network for a certain input.
        '''
        '''
        inputs_list: The inputs going into the neural network.
        '''

        # Converting the input lists into an array with one column and as many rows as
        # there are values, ensuring the functionality of the matrix multiplication.
        inputs = numpy.array(inputs_list, ndmin=2).T

        # The inputs of the hidden layer are the combined and moderated signals, so a matrix
        # multiplication of the weight matrix with the inputs.
        hidden_inputs = numpy.dot(self.wih, inputs)
        # The hidden outputs are the sigmoid function of the hidden inputs.
        hidden_outputs = self.activation_function(hidden_inputs)

        # The inputs of the output layer are the combined and moderated signals, so a matrix
        # multiplication of the weight matrix with the hidden outputs.
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # The final outputs are the sigmoid function of the final inputs.
        final_outputs = self.activation_function(final_inputs)

        # Returning the final outputs of the neural network.
        return final_outputs
