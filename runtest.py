# system libraries
import copy
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pprint
import sklearn
import sklearn.datasets

# user libraries
import pynn

# ===== global flags =====

showGraphs = False
printDebug = True

# ===== global class instances =====

pp = pprint.PrettyPrinter(indent = 4)

# ===== display functionality =====

# plot the decision boundary for predictions made by function fn_predict
def plot_decision_boundary(fn_predict, data):
    # min/max values, plus paddint
    xmin = data.data[:, 0].min() - 0.5
    xmax = data.data[:, 0].max() + 0.5
    ymin = data.data[:, 1].min() - 0.5
    ymax = data.data[:, 1].max() + 0.5

    # generate a grid of datapoints separated by distance delta
    delta = 0.01
    xgrid, ygrid = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))

    # predict labels for generated datapoints
    labels = fn_predict(np.c_[xgrid.ravel(), ygrid.ravel()])
    labels = labels.reshape(xgrid.shape)

    # plot the decision boundary and actual dataset
    pyplot.contourf(xgrid, ygrid, labels, cmap = pyplot.cm.Spectral)
    pyplot.scatter(data.data[:, 0], data.data[:, 1], c = data.labels, cmap = pyplot.cm.Spectral)
    pyplot.show()

# ===== testing functionality =====

# generate and plot a random dataset
def generate_data():
    # generate data
    np.random.seed(32)
    x, y = sklearn.datasets.make_moons(200, noise = 0.20)
    return(pynn.dataset(x, y))

def show_data(data):
    # plot data
    pyplot.scatter(data.data[:, 0], data.data[:, 1], s = 40, c = data.labels, cmap = pyplot.cm.Spectral)
    pyplot.show()

def generate_data_231n():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in xrange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    return(pynn.dataset(X, y))

def test_231n():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in xrange(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    # initialize parameters randomly
    W = 0.01 * np.random.randn(D,K)
    b = np.zeros((1,K))

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3 # regularization strength

    # gradient descent loop
    num_examples = X.shape[0]

    for i in xrange(200):
        # evaluate class scores, [N x K]
        scores = np.dot(X, W) + b 

        # compute the class probabilities
        scoresShifted = copy.deepcopy(scores) #x
        scoresShifted -= np.max(scoresShifted) #x
        exp_scores = np.exp(scoresShifted) #np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[range(num_examples),y])

        data_loss = np.sum(correct_logprobs) / num_examples
        #reg_loss = 0.5*reg*np.sum(W*W)
        loss = data_loss# + reg_loss
        if i % 10 == 0:
            print("====> iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = copy.deepcopy(probs)
        dscores[range(num_examples),y] -= 1
        dscores /= float(num_examples)

        # backpropate the gradient to the parameters (W,b)
        dscoresCopy = copy.deepcopy(dscores)
        dW = np.dot(X.T, dscoresCopy)
        db = np.sum(dscoresCopy, axis=0, keepdims=True)

        dW += reg*W # regularization gradient

        if i % 10 == 0:
            dat2 = np.transpose(copy.deepcopy(X))
            w2 = np.transpose(copy.deepcopy(W))
            b2 = np.transpose(copy.deepcopy(b))
            myscores = np.transpose(pynn.fn_linear(pynn.layerIO(dat2, pynn.grad(w2, b2), None), False).input)
            #print scores
            #print myscores
            print("Score max diff:\t\t" + str(np.max(np.max(scores - myscores))))

            scores2 = copy.deepcopy(scores)
            myprobs = np.transpose(pynn.fn_softmax(pynn.layerIO(np.transpose(scores2), None, None), False).input)
            #print probs
            #print myprobs
            #print str(np.size(probs, 0)) + " " + str(np.size(probs, 1))
            #print str(np.size(myprobs, 0)) + " " + str(np.size(myprobs, 1))
            #print np.max(np.max(np.sum(probs, axis = 1)))
            #print np.max(np.max(np.sum(myprobs, axis = 1)))
            print("Probs total diff:\t" + str(np.sum(np.sum(probs - myprobs))))

            probs2 = copy.deepcopy(probs)
            mydata_loss = np.transpose(pynn.loss_crossEntropy(np.transpose(probs2), y, True).input)
            #print mydata_loss
            #print data_loss
            print("Loss total diff:\t" + str(np.sum(np.sum(data_loss - mydata_loss))))

            probs3 = copy.deepcopy(probs)
            mydscores = np.transpose(pynn.loss_crossEntropy(np.transpose(probs3), y, True).dvOutput)
            print("Dscores total diff:\t" + str(np.sum(np.sum(dscores - mydscores))))

            print("\n")

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db

# ===== main function =====

def test():
    # get data
    data = generate_data()
    #if showGraphs:
    #    show_data(data)

    # train sklearn linear model
    import sklearn.linear_model
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(data.data, data.labels)

    # run example nn on different example dataset
    #test_231n()

    # ===== 1 ======

    if False:

        # create neural net model
        nn = pynn.Model('testnn.json', False)

        # show neural net predictions with no training
        if showGraphs:
            plot_decision_boundary((lambda x: nn.predict(x, True, False)), data)

        # train neural net model
        nn.train(data.data, data.labels, 'hyperparams.json')

        # print predictions for dataset
        #if showGraphs:
        #    print(nn.predict(data.data))

        if showGraphs:
            # plot results
            #plot_decision_boundary((lambda x: clf.predict(x)), data)
            plot_decision_boundary((lambda x: nn.predict(x, True, False)), data)

    # ===== 2 =====

    # another dataset
    data2 = generate_data_231n()

    if False:

        # another network
        nn2 = pynn.Model('test231nn.json', False)

        # show neural net predictions with no training
        if showGraphs:
            plot_decision_boundary((lambda x: nn2.predict(x)), data2)

        # train network
        nn2.train(data2.data, data2.labels, 'hyperparams231.json')

        if showGraphs:
            plot_decision_boundary((lambda x: nn2.predict(x, True, False)), data2)

    # ===== 3 =====

    if False:

        # another network
        nn3 = pynn.Model('test231nn_2.json', False)

        # show neural net predictions with no training
        if showGraphs:
            plot_decision_boundary((lambda x: nn3.predict(x, True, False)), data2)

        # train network
        nn3.train(data2.data, data2.labels, 'hyperparams231.json')

        if showGraphs:
            plot_decision_boundary((lambda x: nn3.predict(x, True, False)), data2)

    # ===== 4 =====

    if True:

        # another network
        nn4 = pynn.Model('test231nn_3.json', True)

        # show neural net predictions with no training
        if showGraphs:
            plot_decision_boundary((lambda x: nn4.predict(x, True, False)), data2)

        # train network
        nn4.train(data2.data, data2.labels, 'hyperparams231.json', True)

        if showGraphs:
            plot_decision_boundary((lambda x: nn4.predict(x, True, False)), data2)

    # ===== 5 =====

    if False:

        # another network
        nn4 = pynn.Model('test231nn_4.json', False)

        # show neural net predictions with no training
        if showGraphs:
            plot_decision_boundary((lambda x: nn4.predict(x, True, False)), data2)

        # train network
        nn4.train(data2.data, data2.labels, 'hyperparams231.json', True)

        if showGraphs:
            plot_decision_boundary((lambda x: nn4.predict(x, True, False)), data2)

def testLoaded():
    data2 = generate_data_231n()

    # initialize network with weights
    nn = pynn.Model('test231nn_3.json')

    # show predictions
    if showGraphs:
        plot_decision_boundary((lambda x: nn.predict(x, True, False)), data2)


def realStuff():
    # ===== real stuff now =====

    np.random.seed(20)

    # get glove data
    gloveData = np.genfromtxt('../glove_dataset/glove_data_2.csv', delimiter=',')
    gloveData = np.append(gloveData, np.genfromtxt('../glove_dataset/glove_data_3.csv', delimiter=','), axis = 0)
    gloveData = np.append(gloveData, np.genfromtxt('../glove_dataset/glove_data_4.csv', delimiter=','), axis = 0)

    # separate labels from data
    # labels are in the last column
    gloveLabels = gloveData[:, np.size(gloveData, 1) - 1].astype(int)
    # get the data by iteslf
    gloveData = gloveData[:, range(0, np.size(gloveData, 1) - 1)]

    # nn for glove
    gloveNN = pynn.Model('glovenn09.json', False)

    # train network
    gloveNN.train(gloveData, gloveLabels, 'hyperparamsGlove09-5.json', False)

    # 17600 64-bit regs

def realLoad():
    # get glove data
    gloveData = np.genfromtxt('../glove_dataset/glove_data_2.csv', delimiter=',')

    # separate labels from data
    # labels are in the last column
    gloveLabels = gloveData[:, np.size(gloveData, 1) - 1].astype(int)
    # get the data by iteslf
    gloveData = gloveData[:, range(0, np.size(gloveData, 1) - 1)]

    # nn for glove
    gloveNN = pynn.Model('glovenn09.json', True)

    gloveData = [133, 65, 29, 91, 141, 83, 145, 130, 151, 145]
    gloveData = np.transpose(np.matrix(gloveData))

    # get accuracy
    predictions = gloveNN.predict(gloveData, False, True, False)

    print(predictions[2])
    print('\n')
    print(predictions[3])
    print('\n')
    print(np.argmax(predictions[3], axis = 0))

#    accuracy = pynn.getAccuracy(predictions, np.transpose(gloveLabels))
#    print("Accuracy:\t" + str(accuracy))

#test()

#testLoaded()

#realStuff()

realLoad()

#pynn.checkWeights('glovenn09-4-nobias-data2.weights')

#pynn.sendWeights('glovenn09-4-nobias.weights', False, True)

#pp.pprint(pynn.readWeights('glovenn05-2.weights'))

