import collections
import copy
import ctypes
import json
import math
import numpy as np
import os.path
import serial
import struct

# ========================
# ===== global state =====
# ========================

# ===== user-set flags =====
useInts = True
twosComplementInputs = False
numBits = 8

# ===== calculated and internal constants =====

# == integer network ==

# scale down the weights so they aren't saturated at initialization
WEIGHT_SCALE = 0.5 * 0.64

# min and max values for weights and biases
maxParam = math.pow(2, numBits - 1) - 1
minParam = - math.pow(2, numBits - 1)

# min and max values for inputs *at every layer interface*
maxInput = math.pow(2, numBits) - 1
minInput = 0
if twosComplementInputs:
    maxInput = math.pow(2, numBits - 1) - 1
    minInput = - math.pow(2, numBits - 1)

# ===== serial interface =====

ser = serial.Serial()
ser.baudrate = 57600
ser.port = 'COM5'
ser.bytesize = serial.EIGHTBITS
ser.stopbits = serial.STOPBITS_ONE

# ==============================
# ===== internal utilities =====
# ==============================

# ===== internal data structures =====

# i/o between layers
layerIO = collections.namedtuple('LayerIO', ['input', 'params', 'dvOutput'])

# gradient for i/o between layers
grad = collections.namedtuple('Gradient', ['w', 'b'])

# layer data for saving
ldat = collections.namedtuple('LayerData', ['ltype', 'params'])

# weights and biases for a layer
class Params:

    def __init__(self, numInputs, numOutputs, params = None):
        # initial params were given
        if params != None:
            self.w = np.matrix(params['w'])
            self.b = np.matrix(params['b'])
            return

        # generate normally distributed random weights
        # scale down for some sense of normalization
        self.w = np.matrix(np.random.randn(numOutputs, numInputs)) / float(np.sqrt(numInputs))
        self.b = np.zeros((numOutputs, 1))

        # scale for integer usage
        if (useInts):
            # scale parameters up
            self.w = np.round(self.w * maxParam * WEIGHT_SCALE)
            self.b = np.round(self.b * maxParam * WEIGHT_SCALE)

            # generate new parameters if the previous parameters were out of range
            self.w[self.w > maxParam] = np.random.randint(minParam, high = maxParam, size = np.size(self.w[self.w > maxParam]))
            self.w[self.w < minParam] = np.random.randint(minParam, high = maxParam, size = np.size(self.w[self.w < minParam]))

# ===== internal functionality =====

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def readWeights(wfilename):
    # check that file exists
    if not os.path.isfile(wfilename):
        raise WeightsError('Could not find weights file!')

    wfile = open(wfilename, 'r')

    # store read-in data
    params = []

    # store loop params
    currentLayer = -1
    currentParamType = 'b'

    # read line-by-line
    for line in wfile:
        # empty read
        if line == '\n':
            continue

        # get divider
        if line[0] == '#':
            # check if this is a new layer
            if (len(line) > 2):
                # update the current layer
                currentLayer += 1

                # add new dictionary for this layer
                params.append({})

                # get the layer type
                params[currentLayer]['type'] = line[1:len(line) - 1]

                # initialize parameters
                currentParamType = 'b'
                params[currentLayer]['w'] = []
                params[currentLayer]['b'] = []

            # otherwise switch whether we are reading weights or biasa
            else:
                currentParamType = 'w'

        # otherwise get data
        else:
            currLine = []
            for x in line.split(','):
                if isfloat(x):
                    currLine.append(float(x))

            params[currentLayer][currentParamType].append(currLine)

    wfile.close()

    return params

# print some useful stats on weights
def checkWeights(wfilename):
    params = readWeights(wfilename)

    # loop variables
    maxWeight = 0;
    minWeight = 0;

    # go through all weights and check values
    for layer in params:
        # we only care about data for linear layers
        if layer['type'] == 'linear':
            # TODO check biases

            # get min weight
            currentMinWeight = np.min(np.min(layer['w']))
            if currentMinWeight < minWeight:
                minWeight = currentMinWeight

            # get max weight
            currentMaxWeight = np.max(np.max(layer['w']))
            if currentMaxWeight > maxWeight:
                maxWeight = currentMaxWeight

    # give output
    print("Max weight: " + str(maxWeight) + "\nMin weight: " + str(minWeight))

# if any layer dimension is greater than 256, or send16 is true, weights are sent as:
# layer number // matrix index i [15:8] // matrix index i [7:0] // matrix index j [15:8] // matrix index j [7:0] // weight value
# otherwise:
# layer number // matrix index i // matrix index j // weight value
def sendWeights(wfilename, send16 = False, test = False):
    if not useInts:
        print('WARNING: pynn is not in integer mode')

    # get everything from the file
    params = readWeights(wfilename)

    # open serial communications
    if not test:
        ser.open()

    # check that serial com worked
    if not test and not ser.is_open:
        raise SerialError('Failed to open serial port!')

    # get the max dimensionality of any layer
    maxDim = 0
    for layer in params:
        if layer['type'] == 'linear':
            currentMaxDim = max([np.size(layer['w'], 0), np.size(layer['w'], 1)])
            if currentMaxDim > maxDim:
                maxDim = currentMaxDim

    # loop variables
    layerNum = 0;

    # send weights over serial
    for layer in params:
        # we only really care about sending data for linear layers
        if layer['type'] == 'linear':
            # increment layer number
            layerNum += 1

            # TODO send biases

            # send weights
            for i in range(0, np.size(layer['w'], 0)):
                for j in range(0, np.size(layer['w'], 1)):
                    if test:
                        print('Packaging weight value ' + str(int(layer['w'][i][j])) + ' cast to ' + str(int(layer['w'][i][j]) & 0xff ))

                    # package data for sending over serial
                    if (maxDim > 256) or send16:
                        #dat = (layerNum, i >> 8, i % 256, j >> 8, j % 256, int(layer['w'][i][j]) & 0xff)
                        dat = (int(layer['w'][i][j]) & 0xff,)
                    else:
                        #dat = (layerNum, i, j, int(layer['w'][i][j]) & 0xff)
                        dat = (int(layer['w'][i][j]) & 0xff,)
                    serializedData = ''
                    for val in dat:
                        serializedData += struct.pack('!B', val)

                    # output/send data
                    if test:
                        print('Actual data: ' + str(layerNum) + ' ' + str(i) + ' ' + str(j) + ' ' + str(int(layer['w'][i][j])))
                        print('Serialized line: ' + str(serializedData))
                    else:
                        ser.write(serializedData)

    # close serial communications
    if not test:
        ser.close()

def fn_linear(IOin, backprop = False):
    # prediction
    # weights * inputs + b
    weightDot = np.matmul(IOin.params.w, IOin.input)
    bias = IOin.params.b * np.ones((np.size(IOin.params.b, 1), np.size(IOin.input, 1)))
    output = weightDot + bias

    # backpropagation
    dvInput = None
    gradient = None
    if backprop:
        # partial for this layer
        # weights' * partial for downstream layer
        dvInput = np.matmul(np.transpose(IOin.params.w), IOin.dvOutput)

        # gradient for each weight/bias this layer
        # partial for this layer * input'
        # (input * weight * downstream error)
        gradW = np.matmul(IOin.dvOutput, np.transpose(IOin.input))

        # treat the bias as another weight + input, where weight = bias, input = 1
        gradB = np.sum(IOin.dvOutput, axis = 1, keepdims = True)

        # package the data together
        gradient = grad(gradW, gradB)

    return layerIO(output, gradient, dvInput)

def fn_rectify(IOin, backprop = False):
    # get necessary parameters
    upperBound = 10
    if useInts:
        upperBound = maxInput

    # prediction
    # use a capped ramp function:  __
    #                           __/
    output = IOin.input
    output[output < 0] = 0
    output[output > upperBound] = upperBound

    # backpropagation
    dvInput = None
    gradient = None
    if backprop:
        # partial for this layer
        dvInput = IOin.dvOutput
        dvInput[output == 0] = 0
        dvInput[output == upperBound] = 0

    return layerIO(output, gradient, dvInput)

def fn_softmax(IOin, backprop = False):
    # prediction
    # shift input values over so the highest value is 0 - do not want infinities
    #>inputShifted = IOin.input
    #>inputShifted -= np.max(inputShifted)

    # scale down data also to prevent infinities
    #>if useInts:
    #>    inputShifted /= float(maxInput * maxInput)

    # calculate softmax: (e^x_i)/(sum over j of (e^x_j))
    # the idea here is to make it such that each vector of probabilities (for
    # each possible classification) is normalized
    #>expInput = np.exp(inputShifted)
    #>output = expInput / np.sum(expInput, axis = 0, keepdims = True)

    # test prediction without actually doing softmax
    # DO NOT USE FOR TRAINING PURPOSES
    output = IOin.input

    # backpropagation
    dvInput = None
    gradient = None
    if backprop:
        # simply push through the error from the cross-entropy loss
        dvInput = IOin.dvOutput

    return layerIO(output, gradient, dvInput)

def loss_crossEntropy(nnOutput, labels, backprop = False):
    # get necessary parameters
    batchSize = np.size(nnOutput, 1)

    # calculate losses
    # note that the cross-entropy cost function is what should be used when a
    # softmax is the last activation function

    # probabilities at the correct classes
    correctLogProbabilities = -np.log(nnOutput[labels, range(batchSize)])

    # average of log probabilities
    loss = (1 / float(batchSize)) * np.sum(correctLogProbabilities)

    # backpropagation
    dvInput = None
    gradient = None
    if backprop:
        # gradient on output scores
        dvInput = nnOutput
        dvInput[labels, range(batchSize)] -= 1
        dvInput = dvInput * (1 / float(batchSize))

    return layerIO(loss, gradient, dvInput)

def getAccuracy(predicted, actual):
    correct = (predicted == actual)
    return np.sum(correct) / float(np.size(correct))

def scaleData(data):
    scaledData = data

    # adjust data to work correctly with ints
    if useInts:
        scaledData /= float(np.max(np.max(scaledData)))
        scaledData = np.round(data * maxInput)
    else:
        scaledData /= float(np.max(np.max(scaledData)))

    return scaledData

# ==============================
# ===== user functionality =====
# ==============================

# ===== data structures =====

# dataset
dataset = collections.namedtuple('Dataset', ['data', 'labels'])

# neural net layer
class Layer:

    def __init__(self, name, ltype, fn, numInputs, numOutputs, params = None):
        # get layer data
        self.name = name
        self.ltype = ltype
        self.fn = fn

        # get weights and bias
        self.params = None
        if numInputs != 0:
            if params != None:
                self.params = Params(numInputs, numOutputs, params)
            else:
                self.params = Params(numInputs, numOutputs)

    # forward pass
    def predict(self, data):
        return self.fn(layerIO(data, self.params, None), False)

    # backpropagation
    def backpropagate(self, data, dvOutput):
        return self.fn(layerIO(data, self.params, dvOutput), True)

    # get params (for saving)
    def getParams(self):
        return ldat(self.ltype, self.params)

    # set params (for backpropagation updates)
    def setParams(self, paramsW, paramsB):
        if self.params != None:
            self.params.w = copy.deepcopy(paramsW)
            self.params.b = copy.deepcopy(paramsB)

    # round params (for integer nn post-processing)
    def roundParams(self):
        if self.params != None:
            self.params.w = np.round(self.params.w)
            self.params.b = np.round(self.params.b)

# neural net model
class Model:

    def __init__(self, modelJsonFile, doLoadWeights = True):
        # read layers json file
        with open(modelJsonFile) as modelFile:
            modelData = json.load(modelFile)

        # extract list of layers
        inputLayerList = modelData['layers']

        # get weight file
        self.weightFile = modelData['weightFile']
        loadedWeights = None
        if os.path.isfile(self.weightFile) and doLoadWeights:
            loadedWeights = readWeights(modelData['weightFile'])

        # initialize layers list
        self.layers = []

        # create layers, add to layer list
        i = 0 # needed for loadedWeights
        for layer in inputLayerList:
            layerFn = None
            numInputs = 0
            numOutputs = 0
            if 'params' in layer:
                numInputs = layer['params']['inputs']
                numOutputs = layer['params']['outputs']

            # choose correct layer function
            if layer['type'] == 'linear':
                layerFn = fn_linear
            elif layer['type'] == 'rectify':
                layerFn = fn_rectify
            elif layer['type'] == 'softmax':
                layerFn = fn_softmax
            else:
                raise ModelError('Invalid layer specified')

            # initialize layer
            newLayer = None
            if loadedWeights != None:
                # error checking
                if loadedWeights[i]['type'] != layer['type']:
                    raise ModelError('Weights file does not match layers!')
                #if numInputs and np.size(loadedWeights[i]['b'], 0) != numInputs:
                #    raise ModelError('Weights file does not match layer size!')
                #if numOutputs and np.size(loadedWeights[i]['w'], 1) != numInputs:
                #    raise ModelError('Weights file does not match layer size!')

                newLayer = Layer(layer['name'], layer['type'], layerFn, numInputs, numOutputs, loadedWeights[i])
            else:
                newLayer = Layer(layer['name'], layer['type'], layerFn, numInputs, numOutputs,)

            # add layer to layers list
            self.layers.append(newLayer)

            i += 1

        # helpful output
        print("Initialized " + str(len(self.layers)) + " layers")

    # ensure dimensions at each layer match up correctly
    def checkDimensions(self, data):
        #TODO
        return

    # forward pass
    def predict(self, data, doScaleData = False, train = False, doTranspose = True):
        if not train:
            # correctly format data
            if np.size(data, 1) < np.size(data, 0) and doTranspose:
                data = np.transpose(data)

        # scale the data to correctly work with the network
        if doScaleData:
            data = scaleData(data)

        # run prediction in each layer
        allData = [data]
        for layer in self.layers:
            layerOut = layer.predict(data)
            data = layerOut.input
            allData.append(data)

        if train:
            # return values as layerIO for backpropagation
            return allData
        else:
            # get location of max value (this is the prediction)
            prediction = np.argmax(data, axis = 0)
            return prediction

    # backpropagation/backwards pass
    def backpropagate(self, data, loss, hyperparams):
        # get useful parts of hyperparams
        lr = hyperparams['learningRate']
        wd = hyperparams['weightDecay']

        # calculate gradients (loop in reverse of forward direction)
        gradW = []
        gradB = []
        for i in range(0, len(self.layers)):
            # get the indicies to loop backwards through the layers
            layerIndex = len(self.layers) - 1 - i

            # get each layer's backpropagation outputs
            layerOut = self.layers[layerIndex].backpropagate(data[layerIndex], loss)

            # store the backpropagation outputs
            loss = layerOut.dvOutput
            if layerOut.params != None:
                gradW.append(layerOut.params.w)
                gradB.append(layerOut.params.b)
            else:
                gradW.append(None)
                gradB.append(None)

        # update parameters (also in reverse order)
        for i in range(0, len(self.layers)):
            # get correct index for layer (we're going backwards)
            layerIndex = len(self.layers) - 1 - i

            # check if layer has any parameters to update
            if self.layers[layerIndex].params != None:
                # updated = old - learning rate * gradient - learning rate * weight decay * old
                # useful notes: http://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
                oldW = self.layers[layerIndex].params.w
                oldB = self.layers[layerIndex].params.b
                newW = oldW - lr * (gradW[i] + wd * oldW)
                newB = oldB# - lr * (gradB[i] + wd * oldB)

                # updates for integer weights
                if useInts:
                    # round new weights
                    #newW = np.round(newW)
                    #newB = np.round(newB)

                    # if weights were too large or small, truncate them
                    newW[newW > maxParam] = maxParam
                    newW[newW < minParam] = minParam
                    newB[newB > maxParam] = maxParam
                    newW[newW < minParam] = minParam

                # actually set the parameters in the layer
                self.layers[layerIndex].setParams(newW, newB)

        return

    # save trained weights
    def save(self):
        # open file for writing
        wfile = open(self.weightFile, 'w')

        # get each layer's data and write it
        for layer in self.layers:
            dat = layer.getParams()

            # write separator
            wfile.write('#')

            # write layer type
            wfile.write(dat.ltype)
            wfile.write('\n')

            # write parameters if necessary
            if dat.params != None:
                # write b parameter
                for row in dat.params.b:
                    wfile.write(str(float(row.item(0))))
                    wfile.write('\n')

                # write separator
                wfile.write('#\n')

                # write w parameter
                for row in dat.params.w:
                    for i in range(0, row.size):
                        wfile.write(str(row.item(i)))
                        wfile.write(',')
                    wfile.write('\n')

        wfile.close()

    # training
    def train(self, data, labels, hyperparamsFilePath = None, doScaleData = False):
        # ===== pre-processing =====

        # correctly format data
        if np.size(data, 1) < np.size(data, 0):
            data = np.transpose(data)
            labels = np.transpose(labels)

        # scale the data to correctly work with the network
        if doScaleData:
            data = scaleData(data)

        # sanity check
        print("Max input value:" + str(np.max(np.max(np.max(data)))))
        print("Min input value: " + str(np.min(np.min(np.min(data)))))

        # get default hyperparameters if no file is specified
        hyperparams = None
        if hyperparamsFilePath == None :
            hyperparams = {
                'momentum': 0.95,
                'learningRateDecay': 0.98,
                'learningRate': 0.01,
                'weightDecay': 0.0005,
                'batchSize': 128,
                'iterations': 10000
            }
        elif not os.path.isfile(hyperparamsFilePath):
            raise TrainError('Sepcified hyperparams file does not exist!')

        # read hyperparameters json file
        with open(hyperparamsFilePath) as hyperparamsFile:
            hyperparams = json.load(hyperparamsFile)
            print("Read hyperparameters from file")

        # ==== generate minibatches =====

        # randomize indices for data
        indices = range(0, np.size(data, 1))
        randomizedIndices = np.random.choice(indices, size = np.size(data,1), replace = False)

        # randomize actual data
        randomizedData = data[:, randomizedIndices]
        randomizedLabels = labels[randomizedIndices]

        # get batches
        batches = []
        batchesLabels = []
        numBatches = int(np.floor(np.size(data, 1) / float(hyperparams['batchSize'])))
        for i in range(0, numBatches):
            # get indices for batches
            minBatchIndex = i * hyperparams['batchSize']
            maxBatchIndex = (i + 1) * hyperparams['batchSize'] - 1
            batchIndices = range(minBatchIndex, maxBatchIndex)

            # get a batch
            batches.append(randomizedData[:, batchIndices])
            batchesLabels.append(randomizedLabels[batchIndices])

        # get the last batch
        batchIndices = range(numBatches * hyperparams['batchSize'], np.size(data, 1))
        if len(batchIndices) != 0:
            batches.append(randomizedData[:, batchIndices])
            batchesLabels.append(randomizedLabels[batchIndices])

        # ===== train the network =====

        for i in range(0, hyperparams['iterations']):
            # select a random mini batch from the data
            batchIndex = np.random.randint(0, len(batches))
            batch = batches[batchIndex]
            batchLabels = batchesLabels[batchIndex]

            # ===== forward pass =====

            # get output of every layer
            allOutput = self.predict(batch, False, True)

            # get output of only the last layer (the actual output)
            output = allOutput[len(allOutput) - 1]
            predictions = np.argmax(output, axis = 0)

            # get losses
            lossOutput = loss_crossEntropy(output, batchLabels, True)

            # ===== backwards pass =====

            # backpropagation
            self.backpropagate(allOutput, lossOutput.dvOutput, hyperparams)

            # update hyperparameters
            hyperparams['learningRate'] *= hyperparams['learningRateDecay']
            hyperparams['weightDecay'] *= hyperparams['learningRateDecay']

            # ===== generate output for user =====

            # print training progress
            if (i % 100) == 0:
                accuracy = getAccuracy(predictions, batchLabels)
                print("Iteration: \t" + str(i) + "\tAccuracy:\t" + "{0:.6f}".format(accuracy) + "\tLoss:\t" + "{0:.6f}".format(lossOutput.input))

        # ===== final processing =====

        print("Training complete!")

        # get accuracy for entire dataset
        predictions = self.predict(data, False, False)
        accuracy = getAccuracy(predictions, labels)
        print("Overall accuracy:\t" + str(accuracy))

        # integer neural net post-processing
        if useInts:
            print("Integer post-processing . . .")

            # round weights to nearest whole numbers
            for layer in self.layers:
                layer.roundParams()

            # get accuracy for entire dataset after roundng
            predictions = self.predict(data, False, False)
            accuracy = getAccuracy(predictions, labels)
            print("Overall accuracy:\t" + str(accuracy))

        # ===== save relevant data =====

        # save the trained model at the specified path
        self.save()

