# Theo Satloff and Walker Griggs
# Artificial Neural Network (Autoencoder)
# May 2016

# sources ------------
# http://www.webpages.ttu.edu/dleverin/neural_network/neural_networks.html
# https://www.willamette.edu/~gorr/classes/cs449/momrate.html
# http://cs.colby.edu/courses/S16/cs251/LectureNotes/Lecture_33_NeuralNetworks_04_25_16.pdf
# http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
# http://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325
# http://ufldl.stanford.edu/tutorial/
# http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
# http://neuralnetworksanddeeplearning.com/chap3.html
# https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Activation_Functions

import random
import time
import csv
import numpy as np
np.seterr(all = 'ignore')

from heapq import nsmallest
from termcolor import colored
from tabulate import tabulate
import ActivationFunctions

# Artifical Neural Network Parent Class
class ANN(object):
    def __init__(self, input, hidden, output, iterations = 50, learningRate = 0.01, momentum = 0, decayRate = 0, activationType = 'sigmoid', backpropType = 'online'):

        """ Create a standard, single layer neural network

        Keyword arguments:
            input -- # of input nodes
            hidden -- # of hidden nodes
            output -- # of output nodes
            iterations -- number iterations through backprop and feed forward (default: 50)
            learningRate -- influence of each iteration on the weight(default: 1%)
            momentum -- persistence of each iteraiton upon meeting a local minima (default: 0)
            decayRate -- rate at which the weights to exponentially decay to zero (default: 0)
            activationType -- choice between activation functions
            backpropType -- choice between backpropigation functions (default: online)
        """

        #Iinitialize parameters
        self.iterations = iterations
        self.learningRate = learningRate
        self.momentum = momentum
        self.decayRate = decayRate
        self.activationType = activationType
        self.backpropType = backpropType

        # Initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # Set up array of 1s for activations
        self.inputLayer = np.ones(self.input)
        self.hiddenLayer = np.ones(self.hidden)
        self.outputLayer = np.ones(self.output)

        #Create randomized weights (gaussian distribution)
        inputRange = 1.0 / self.input ** (1/2)
        self.inputWeight = np.random.normal(loc = 0, scale = inputRange, size = (self.input, self.hidden))
        self.outputWeight = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)

        # Create arrays of change 0s (0s matrx)
        self.inputChange = np.zeros((self.input, self.hidden))
        self.outputChange = np.zeros((self.hidden, self.output))

    # Feed Forward Epoch
    def feedForward(self, inputs):
        """
        Update weights for each note moving backwards throught the network

        1) Loop over all of the nodes in the hidden layer. Add outputs from input layer * weights.
        2) Loop over all nodes in the output layer. Add outputs from the hidden layer * weights.
        3) Apply the training patterns.
        4) Apply the weight changes to the network, modified by a learning rate.
        """

        # Check to see if Inputs match
        if len(inputs) != self.input-1:
            raise ValueError('Wrong number of inputs you silly goose!')

        # Input Activations
        self.inputLayer[0:self.input -1] = inputs

        # Hidden Activations
        sum = np.dot(self.inputWeight.T, self.inputLayer)
        self.hiddenLayer = tanh(sum)

        # Output Activations
        sum = np.dot(self.outputWeight.T, self.hiddenLayer)
        if self.activationType == 'sigmoid':
            self.outputLayer = sigmoid(sum)
        elif self.activationType == 'softmax':
            self.outputLayer = softmax(sum)
        elif self.activationType == 'elliot':
            self.outputLayer = elliot(sum)
        else:
            raise ValueError('Please choose an implemented ativation function')

        return self.outputLayer

    # Back Propogate Epoch
    def backPropagate(self, targets):
        """"
        Update weights for each note moving backwards throught the network

        output layer:
        1) Get difference between output and target.
        2) Get slope of sigmoid to get weight change
        3) Update weights based on learning rate and dsigmoid

        hidden layer:
        1) Sum (output weights * amount change)
        2) Derive to calculate amount of change
        3) Change weights based on learning rate, derivative, and decay.
        """

        # Check to see if Outputs match
        if len(targets) != self.output:
            raise ValueError('Wrong number of inputs')

        # Error for Output
        # The delta tells you which direction to change the weights  (generalized delta rule)
        if self.activationType == 'sigmoid' or self.activationType == 'elliot':
            outputDeltas = dsigmoid(self.outputLayer) * -(targets - self.outputLayer)
        elif self.activationType == 'softmax':
            outputDeltas = -(targets - self.outputLayer)
        else:
            raise ValueError('choose one of the activation functions plz')

        # Error for Hidden
        error = np.dot(self.outputWeight, outputDeltas)
        hiddenDeltas = dtanh(self.hiddenLayer) * error

        # Update Hidden to Output weights
        change = outputDeltas * np.reshape(self.hiddenLayer, (self.hiddenLayer.shape[0],1))
        self.outputWeight -= self.learningRate * change + self.outputChange * self.momentum
        self.outputChange = change

        # Update Hidden to Input weights
        change = hiddenDeltas * np.reshape(self.inputLayer, (self.inputLayer.shape[0], 1))
        self.inputWeight -= self.learningRate * change + self.inputChange * self.momentum
        self.inputChange = change

        # Update Input to Hidden weights
        if self.activationType == 'sigmoid' or self.activationType == 'elliot':
            error = sum(0.5 * (targets - self.outputLayer)**2)
        elif self.activationType == 'softmax':
            error = -sum(targets * np.log(self.outputLayer))

        return error


    # Train the ANN
    def train(self, patterns):
        """"
        Train the Artifical Neural Network in accordance with data set patterns.

        1) call feedForward, that gives output with randomized weights.
        2) call backPropagate, tunes the weights to give better prediction.
        3) call feedForward again, using the updated weights from step 2.
        4) repeat until the error gets closer to 0.
        """"

        start = time.time()

        print 'Activation Type: ', str(self.activationType).upper()

        # Gradient Descent Selection
        if self.backpropType == "stochastic":
            iterator = np.random.permutation(self.iterations)
        elif self.backpropType == "online":
            iterator = range(self.iterations)

        # Backprop loop
        for i in iterator:
            error = 0.0
            random.shuffle(patterns)
            for p in patterns:
                inputs = p[0]
                targets = p[1]

                self.feedForward(inputs)
                error += self.backPropagate(targets)

            # Code to write each error to CSV row.
            # with open('error.txt', 'a') as errorfile:
            #     errorfile.write(str(error) + ',')
            #     errorfile.close()

            #  print error
            if i % 10 == 0:
                error = error/np.shape(patterns)[0]
                print('Training error %-.5f' % error)

            # learning rate decay
            self.learningRate = self.learningRate * (self.learningRate / (self.learningRate + (self.learningRate * self.decayRate)))

        # Drops CSV row to new line.
        # with open('error.txt', 'a') as errorfile:
        #     errorfile.write('\n')
        #     errorfile.close()

        # Print train time.
        end = time.time()
        curTime = "%-.5s seconds" % (end-start)
        print colored(curTime, 'cyan')

    # Test the newly trained data set.
    def test(self, data, s):
        """ Prints out the target, normalized value, unnormalized value, and prediction status. """

        right = 0
        wrong = 0

        for part in data:
            value = self.feedForward(part[0])[0]
            final = nsmallest(1, s, key=lambda x: abs(x-value))

            # Wrong
            if part[1][0] != final[0]:
                print part[1], '->', ('%-.6f' % value), ' ', colored("[ WRONG ]", 'red'), final[0]
                wrong += 1

            # Right
            else:
                print part[1], '->', ('%-.6f' % value), ' ', colored("[ RIGHT ]", 'green'), final[0]
                right += 1

        print colored(str(right)+'/'+str(right+wrong), 'blue')


    # Save ANN to file.
    def save(self, fn):
        masterList = [["Input"],["Input Weight"],["Hidden"],["Output"],["Output Weight"]]

        # Save Input Layer
        for node in self.inputLayer:
            masterList[0].append(node)

        # Save Input Weight
        masterList[1].append(self.inputWeight.shape[0]) # number of rows
        masterList[1].append(self.inputWeight.shape[1]) # number of columns
        for weight in self.inputWeight.tolist():
            for each in weight:
                masterList[1].append(each)

        # Save Hidden Layer
        for node in self.hiddenLayer:
            masterList[2].append(node)

        # Save Output Layer
        for node in self.outputLayer:
            masterList[3].append(node)

        # Save Output Weight
        masterList[4].append(self.outputWeight.shape[0]) # number of rows
        masterList[4].append(self.outputWeight.shape[1]) # number of columns
        for weight in self.outputWeight.tolist():
            for each in weight:
                masterList[4].append(each)

        # Write to filename provided
        values = open(fn, 'wb')
        wr = csv.writer(values, quoting=csv.QUOTE_ALL)
        for row in masterList:
            wr.writerow(row)

    # Read in trained network.
    def readTrain(self, fn):
        f = open(fn)
        rows = csv.reader(f)

        for i, row in enumerate(rows):

            # Read Input Layer
            if i == 0:
                self.inputLayer = np.array(map(float,row[1:]))

            # Read Input Weight
            elif i == 1:
                rows = map(float, row[3:])
                self.inputWeight = np.array([rows[i:i+int(row[2])] for i in range(0, len(rows), int(row[2]))]) # unflatten list from write out

            # Read Hidden Layer
            elif i == 2:
                self.hiddenLayer = np.array(map(float,row[1:]))

            # Read Output Layer
            elif i == 3:
                self.outputLayer = np.array(map(float,row[1:]))

            # Read Output Weight
            elif i == 4:
                rows = map(float, row[3:])
                self.outputWeight = np.array([rows[i:i+int(row[2])] for i in range(0, len(rows), int(row[2]))]) # unflatten list from write out.

# Run the ANN.
def run():
    global inputCols
    global inputList
    global outputCols

    def readIn(input, train = True):
        global inputCols
        global inputList
        global outputCols

        input = np.loadtxt(input, delimiter = ',')
        #output = np.loadtxt(output, delimiter = ',')

        inputCols = input.shape[1]-1 # right now this is hardcoded for 1 output

        data = input[:,:-1]
        targets = input[:,-1:]
        outputCols = targets.shape[1]

        data -= data.min() # scale the data so values are between 0 and 1
        data /= data.max() # scale

        targets -= targets.min() # scale the data so values are between 0 and 1
        targets /= targets.max() # scale

        inputList = np.unique(targets) # list of codes to be used for prediction later on

        patterns = []

        # populate the tuple list with the data
        for i in range(data.shape[0]):
            full = list((data[i,:].tolist(), targets[i].tolist()))
            patterns.append(full)

        return patterns

    train = readIn('datasets/iris_proj8_train-1.csv') #UCI-X-train-1
    tictac = ANN(inputCols, 10, outputCols, iterations = 100, learningRate = 0.01, momentum = 0.5, decayRate = 0.0001, activationType = 'elliot', backpropType = 'online')

    tictac.train(train)

    #test = readIn('datasets/iris_proj8_test-1.csv')

    def testPredictions():
        test = readIn('datasets/iris_proj8_test-1.csv')

        # run tests
        tictac.test(test, inputList)

    def testReadIn():
        test = readIn('datasets/iris_proj8_test-1.csv')

        # write out train file to be read in later
        tictac.save('results/values.csv')

        # read in train file
        tictac.readTrain('results/values.csv')

        # run tests on train file
        tictac.test(test, inputList)


    # --------- Uncomment to test File Read --------- #
    #testReadIn()

    # --------- Uncomment to test Predictions --------- #
    #testPredictions()
if __name__ == '__main__':
    run()
