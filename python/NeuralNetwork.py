import numpy as np
import random

get_ipython().magic("run 'Data_Munging.ipynb'")

class Network(object): 
    
    def __init__(self,sizes): # sizes a vector containing the number of neurons in the respective layers
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, a):# return the ouput of the network given a the input
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None): 
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data: 
                print "Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data),n_test)
            else: 
                print "Epoch {0} complete".format(j)
                
    def update_mini_batch(self, mini_batch,eta): 
        '''Update the network's biases and weights by applying gradient dexcent using backpropagation to a single mini batch''' 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        
        
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        '''Return the number of test inputs for which the neural network outputs the correct result.'''
        test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        '''Return the vector of partial derivatives \partial C_x / \partial a for the output activations. '''
        return (output_activations - y)
    

def sigmoid(z): 
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    '''Derivative of the sigmoid function'''
    return sigmoid(z)*(1-sigmoid(z))



dayList = [sensorData['Start time'][x].day for x in xrange(1,len(sensorData))]
dayList= list(set(dayList))
l = list(activityLabel.index)
errorRate = 0
# confusionMatrix = confusion_matrix(activityLabel, activityLabel, labels = l)
# confusionMatrix = confusionMatrix - confusionMatrix #we have then a 0 confusion matrix

testingDay = 25
# while testingDay == 26:
#     randomNumber = randint(0,len(dayList)-1)
#     testingDay = dayList[randomNumber]
# print testingDay

trainingSensor = [sensorData['Start time'][x].day != testingDay and sensorData['End time'][x].day != testingDay for x in range(1,len(sensorData))]
trainingSensor = trainingSensor + [False]
trainingSensorData = sensorData[trainingSensor]
trainingSensorData.index = np.arange(1,len(trainingSensorData)+1)

trainingActivity = [activityData['Start time'][x].day != testingDay and activityData['End time'][x].day != testingDay for x in range(1,len(activityData))]
trainingActivity = trainingActivity + [False]
trainingActivityData = activityData[trainingActivity]
trainingActivityData.index = np.arange(1,len(trainingActivityData)+1)

trainingFeatureMatrix, trainingLabel = convert2LastFiredFeatureMatrix(trainingSensorData,trainingActivityData, 60)
cumuSensor, cumuActivity = cumulationTable(trainingFeatureMatrix, trainingLabel)
normalizedCumuSensor = [x/sum(x) for x in cumuSensor]
lst = []
for s in normalizedCumuSensor:
    lst2 = [[x] for x in s]
    lst.append(lst2)

normalizedCumuSensor = np.asarray(lst)

labelTrain = []
for x in xrange(len(cumuActivity)):
    labelTrain.append([[0] for i in xrange(len(activityLabel))])
    for a,b in enumerate(activityLabel.index):
        if b == cumuActivity[x]:
            labelTrain[x][a] = [1]


labelTrainingData = np.asarray(labelTrain)

trainingData = zip(normalizedCumuSensor, labelTrainingData)


daySensor = [sensorData['Start time'][x].day == testingDay and sensorData['End time'][x].day == testingDay for x in range(1,len(sensorData))]
daySensor = daySensor + [False]
daySensorData = sensorData[daySensor]
daySensorData.index = np.arange(1,len(daySensorData)+1)

dayActivity = [activityData['Start time'][x].day == testingDay and activityData['End time'][x].day == testingDay for x in range(1,len(activityData))]
dayActivity = dayActivity + [False]
dayActivityData = activityData[dayActivity]
dayActivityData.index = np.arange(1,len(dayActivityData)+1)

testingFeatureMatrix, testingLabel = convert2LastFiredFeatureMatrix(daySensorData,dayActivityData,60)
cumuSen, cumuAct = cumulationTable(testingFeatureMatrix, testingLabel)
normalizedCumuSen = [x/sum(x) for x in cumuSen]
lst = []
for s in normalizedCumuSen:
    lst2 = [[x] for x in s]
    lst.append(lst2)
normalizedCumuSen = np.asarray(lst)

labelTestingData = np.asarray(cumuAct)

testingData = zip(normalizedCumuSensor, labelTestingData)


net = Network([14,10,8])

net.SGD(trainingData,150,30,2.0, test_data = testingData)

len(trainingData)



import neurolab as nl



















