from numpy import exp, array, random, dot

debug = False

class NeuralNetwork():
    def __init__(self, seedValue=1):
#         see the RNG (potentially use gpu here for RNG seeding)
        global debug
        random.seed(seedValue)

        #  model a single neuron, with 3 inputs and 1 output
        #  assigning random weights to a 3x1 matrix, with values from -1 to 1
        #  with a mean of 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1

        if (debug):
            print self.synaptic_weights

    # the sigmoid function (s shaped), passing the weighted sum of the input though
    # this function to normalise them between 0 and 1
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    # gradient of sigmoid function
    @staticmethod
    def sigmoid_deriavative(x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for interation in xrange(number_of_training_iterations):
            # pass the training set through the neural network
            output = self.predict(training_set_inputs)

            # calculate the error
            error = training_set_outputs - output
            calibration = dot(training_set_inputs.T, error * self.sigmoid_deriavative(output))

    def predict(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))



# use of NN class
if __name__ == '__main__':

    # init a single neuron
    neural_network = NeuralNetwork(seedValue=1)

    print 'Random synaptic weights:'
    print neural_network.synaptic_weights

    # training set, 4 examples each consisting of 3 input values and 1 output
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0,1,1,0]]).T

    #train the neural network using the training set
    # do it 10000 times for good measure, making small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print 'New synaptic weights after training:'
    print neural_network.synaptic_weights

    # Test the neural network
    print 'predicting:'
    print neural_network.predict(array)
