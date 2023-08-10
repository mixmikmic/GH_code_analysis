
def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
   """Produce a one-hot-encoding from a list of features and an OHE dictionary.

   Note:
       You should ensure that the indices used to create a SparseVector are sorted.

   Args:
       rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
           feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
       OHEDict (dict): A mapping of (featureID, value) to unique integer.
       numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
           value).

   Returns:
       SparseVector: A SparseVector of length numOHEFeats with indicies equal to the unique
           identifiers for the (featureID, value) combinations that occur in the observation and
           with values equal to 1.0.
           
   """
   rawFeats = sorted(rawFeats, key=lambda x: x[0])  
   return SparseVector(numOHEFeats, [(OHEDict[(featID, value)],1) for (featID, value) in rawFeats])


def createOneHotDict(inputData):
   """Creates a one-hot-encoder dictionary based on the input data.

   Args:
       inputData (RDD of lists of (int, str)): An RDD of observations where each observation is
           made up of a list of (featureID, value) tuples.

   Returns:
       dict: A dictionary where the keys are (featureID, value) tuples and map to values that are
           unique integers.
   """
   X = (inputData.flatMap(lambda x: x).distinct())
   Y = (X.zipWithIndex().collectAsMap())
   return Y

import os.path
baseDir = os.path.join('data')
inputPath = os.path.join('cs190', 'dac_sample.txt')
fileName = os.path.join(baseDir, inputPath)

if os.path.isfile(fileName):
    rawData = (sc
               .textFile(fileName, 2)
               .map(lambda x: x.replace('\t', ',')))  # work with either ',' or '\t' separated data
    print rawData.take(1)

weights = [.8, .1, .1]
seed = 42
# Use randomSplit with weights and seed
rawTrainData, rawValidationData, rawTestData = rawData.randomSplit(weights, seed)
# Cache the data
rawTrainData.cache()
rawValidationData.cache()
rawTestData.cache()
nTrain = rawTrainData.count()
nVal = rawValidationData.count()
nTest = rawTestData.count()
print nTrain, nVal, nTest, nTrain + nVal + nTest
print rawData.take(1)


def parsePoint(point):
   """Converts a comma separated string into a list of (featureID, value) tuples.

   Note:
       featureIDs should start at 0 and increase to the number of features - 1.

   Args:
       point (str): A comma separated string where the first value is the label and the rest
           are features.

   Returns:
       list: A list of (featureID, value) tuples.
   """
   x="".join(point).split(',') 
       
   return [(i,x[i+1]) for i in np.arange(len(x)-1)]

parsedTrainFeat = rawTrainData.map(parsePoint)

numCategories = (parsedTrainFeat
                .flatMap(lambda x: x)
                .distinct()
                .map(lambda x: (x[0], 1))
                .reduceByKey(lambda x, y: x + y)
                .sortByKey()
                .collect())

print numCategories[2][1]


ctrOHEDict = createOneHotDict(parsedTrainFeat)
numCtrOHEFeats = len(ctrOHEDict.keys())
print numCtrOHEFeats
print ctrOHEDict[(0, '')]

from pyspark.mllib.regression import LabeledPoint


def parseOHEPoint(point, OHEDict, numOHEFeats):
   """Obtain the label and feature vector for this raw observation.

   Note:
       You must use the function `oneHotEncoding` in this implementation or later portions
       of this lab may not function as expected.

   Args:
       point (str): A comma separated string where the first value is the label and the rest
           are features.
       OHEDict (dict of (int, str) to int): Mapping of (featureID, value) to unique integer.
       numOHEFeats (int): The number of unique features in the training dataset.

   Returns:
       LabeledPoint: Contains the label for the observation and the one-hot-encoding of the
           raw features based on the provided OHE dictionary.
       
   """
   x= "".join(point).split(',') 
   rawFeats = parsePoint(point)
  
   lab=x[0]
   feat = oneHotEncoding(rawFeats, OHEDict, numOHEFeats)
   return LabeledPoint(float(lab), feat)

def bucketFeatByCount(featCount):
    """Bucket the counts by powers of two."""
    for i in range(11):
        size = 2 ** i
        if featCount <= size:
            return size
    return -1

featCounts = (OHETrainData
              .flatMap(lambda lp: lp.features.indices)
              .map(lambda x: (x, 1))
              .reduceByKey(lambda x, y: x + y))
featCountsBuckets = (featCounts
                     .map(lambda x: (bucketFeatByCount(x[1]), 1))
                     .filter(lambda (k, v): k != -1)
                     .reduceByKey(lambda x, y: x + y)
                     .collect())
print featCountsBuckets

import matplotlib.pyplot as plt

x, y = zip(*featCountsBuckets)
x, y = np.log(x), np.log(y)

def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(4, 14, 2))
ax.set_xlabel(r'$\log_e(bucketSize)$'), ax.set_ylabel(r'$\log_e(countInBucket)$')
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
pass


def oneHotEncoding(rawFeats, OHEDict, numOHEFeats):
   """Produce a one-hot-encoding from a list of features and an OHE dictionary.

   Note:
       If a (featureID, value) tuple doesn't have a corresponding key in OHEDict it should be
       ignored.

   Args:
       rawFeats (list of (int, str)): The features corresponding to a single observation.  Each
           feature consists of a tuple of featureID and the feature's value. (e.g. sampleOne)
       OHEDict (dict): A mapping of (featureID, value) to unique integer.
       numOHEFeats (int): The total number of unique OHE features (combinations of featureID and
           value).

   Returns:
       SparseVector: A SparseVector of length numOHEFeats with indicies equal to the unique
           identifiers for the (featureID, value) combinations that occur in the observation and
           with values equal to 1.0.
   """
   return SparseVector(numOHEFeats, [(OHEDict[(featID, value)],1) for (featID, value) in set(rawFeats).intersection(OHEDict.keys())])

from pyspark.mllib.classification import LogisticRegressionWithSGD

# fixed hyperparameters
numIters = 50
stepSize = 10.
regParam = 1e-6
regType = 'l2'
includeIntercept = True


model0 = LogisticRegressionWithSGD.train(OHETrainData , iterations=numIters, step=stepSize, regParam=regParam, regType=regType, intercept =includeIntercept)
sortedWeights = sorted(model0.weights)
print sortedWeights[:5], model0.intercept


from math import log

def computeLogLoss(p, y):
   """Calculates the value of log loss for a given probabilty and label.

   Note:
       log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
       and when p is 1 we need to subtract a small value (epsilon) from it.

   Args:
       p (float): A probabilty between 0 and 1.
       y (int): A label.  Takes on the values 0 and 1.

   Returns:
       float: The log loss value.
   """
   epsilon = 10e-12
   a=sorted([p,epsilon,1-epsilon])
   return -log((1-y)-(1-2*y)*a[1]) 


# Note that our dataset has a very high click-through rate by design
# In practice click-through rate can be one to two orders of magnitude lower
classOneFracTrain = OHETrainData.map(lambda x: x.label).reduce(lambda x,y: x+y)/OHETrainData.count()
print classOneFracTrain

logLossTrBase =  OHETrainData.map(lambda x: computeLogLoss(classOneFracTrain , x.label)).reduce(lambda x,y: x+y)/OHETrainData.count()
print 'Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase)


from math import exp #  exp(-t) = e^-t

def getP(x, w, intercept):
   """Calculate the probability for an observation given a set of weights and intercept.

   Note:
       We'll bound our raw prediction between 20 and -20 for numerical purposes.

   Args:
       x (SparseVector): A vector with values of 1.0 for features that exist in this
           observation and 0.0 otherwise.
       w (DenseVector): A vector of weights (betas) for the model.
       intercept (float): The model's intercept.

   Returns:
       float: A probability between 0 and 1.
   """
   rawPrediction =w.dot(x) + intercept
   
   rawPrediction = min(rawPrediction, 20)
   rawPrediction = max(rawPrediction, -20)
   return 1/(1+exp(-rawPrediction))

trainingPredictions =  OHETrainData.map(lambda x: getP(x.features, model0.weights,model0.intercept))

print trainingPredictions.take(5)


def evaluateResults(model, data):
   """Calculates the log loss for the data given the model.

   Args:
       model (LogisticRegressionModel): A trained logistic regression model.
       data (RDD of LabeledPoint): Labels and features for each observation.

   Returns:
       float: Log loss for the data.
   """
   prob = data.map(lambda x: getP(x.features, model.weights,model.intercept)).collect()
   lab= data.map(lambda x: x.label).collect()
   L=len(lab)
   b=[computeLogLoss(prob[i],lab[i]) for i in np.arange(L)]
   return sum(b)/L

logLossTrLR0 = evaluateResults(model0, OHETrainData)
print ('OHE Features Train Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
      .format(logLossTrBase, logLossTrLR0))

logLossValBase = OHEValidationData.map(lambda x: computeLogLoss(classOneFracTrain , x.label)).reduce(lambda x,y: x+y)/OHEValidationData.count()


logLossValLR0 = evaluateResults(model0, OHEValidationData)
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, logLossValLR0))

labelsAndScores = OHEValidationData.map(lambda lp:
                                            (lp.label, getP(lp.features, model0.weights, model0.intercept)))
labelsAndWeights = labelsAndScores.collect()
labelsAndWeights.sort(key=lambda (k, v): v, reverse=True)
labelsByWeight = np.array([k for (k, v) in labelsAndWeights])

length = labelsByWeight.size
truePositives = labelsByWeight.cumsum()
numPositive = truePositives[-1]
falsePositives = np.arange(1.0, length + 1, 1.) - truePositives

truePositiveRate = truePositives / numPositive
falsePositiveRate = falsePositives / (length - numPositive)

# Generate layout and plot data
fig, ax = preparePlot(np.arange(0., 1.1, 0.1), np.arange(0., 1.1, 0.1))
ax.set_xlim(-.05, 1.05), ax.set_ylim(-.05, 1.05)
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
plt.plot(falsePositiveRate, truePositiveRate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.plot((0., 1.), (0., 1.), linestyle='--', color='#d6ebf2', linewidth=2.)  # Baseline model
pass

from collections import defaultdict
import hashlib

def hashFunction(numBuckets, rawFeats, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        numBuckets (int): Number of buckets to use as features.
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)


def parseHashPoint(point, numBuckets):
   """Create a LabeledPoint for this observation using hashing.

   Args:
       point (str): A comma separated string where the first value is the label and the rest are
           features.
       numBuckets: The number of buckets to hash to.

   Returns:
       LabeledPoint: A LabeledPoint with a label (0.0 or 1.0) and a SparseVector of hashed
           features.
   """
   
   x= "".join(point).split(',') 
   rawFeats = parsePoint(point)
  
   lab=x[0]
   feat = hashFunction(numBuckets, rawFeats  , printMapping=False)
   return LabeledPoint(float(lab), SparseVector(numBuckets, feat))

numBucketsCTR = 2 ** 15
hashTrainData = rawTrainData.map(lambda x: parseHashPoint(x,numBucketsCTR))
hashTrainData.cache()
hashValidationData = rawValidationData.map(lambda x: parseHashPoint(x,numBucketsCTR))
hashValidationData.cache()
hashTestData = rawTestData.map(lambda x: parseHashPoint(x,numBucketsCTR))
hashTestData.cache()

print hashTrainData.take(1)

numIters = 500
regType = 'l2'
includeIntercept = True

# Initialize variables using values from initial model training
bestModel = None
bestLogLoss = 1e10


stepSizes = [1,10]
regParams = [1e-6,1e-3]
for stepSize in stepSizes:
   for regParam in regParams:
       model = (LogisticRegressionWithSGD
                .train(hashTrainData, numIters, stepSize, regParam=regParam, regType=regType,
                       intercept=includeIntercept))
       logLossVa = evaluateResults(model, hashValidationData)
       print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
              .format(stepSize, regParam, logLossVa))
       if (logLossVa < bestLogLoss):
           bestModel = model
           bestLogLoss = logLossVa

print ('Hashed Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
      .format(logLossValBase, bestLogLoss))

from matplotlib.colors import LinearSegmentedColormap

# Saved parameters and results.  Eliminate the time required to run 36 models
stepSizes = [3, 6, 9, 12, 15, 18]
regParams = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
logLoss = np.array([[ 0.45808431,  0.45808493,  0.45809113,  0.45815333,  0.45879221,  0.46556321],
                    [ 0.45188196,  0.45188306,  0.4518941,   0.4520051,   0.45316284,  0.46396068],
                    [ 0.44886478,  0.44886613,  0.44887974,  0.44902096,  0.4505614,   0.46371153],
                    [ 0.44706645,  0.4470698,   0.44708102,  0.44724251,  0.44905525,  0.46366507],
                    [ 0.44588848,  0.44589365,  0.44590568,  0.44606631,  0.44807106,  0.46365589],
                    [ 0.44508948,  0.44509474,  0.44510274,  0.44525007,  0.44738317,  0.46365405]])

numRows, numCols = len(stepSizes), len(regParams)
logLoss = np.array(logLoss)
logLoss.shape = (numRows, numCols)

fig, ax = preparePlot(np.arange(0, numCols, 1), np.arange(0, numRows, 1), figsize=(8, 7),
                      hideLabels=True, gridWidth=0.)
ax.set_xticklabels(regParams), ax.set_yticklabels(stepSizes)
ax.set_xlabel('Regularization Parameter'), ax.set_ylabel('Step Size')

colors = LinearSegmentedColormap.from_list('blue', ['#0022ff', '#000055'], gamma=.2)
image = plt.imshow(logLoss,interpolation='nearest', aspect='auto',
                    cmap = colors)
pass


# Log loss for the best model from (5d)
logLossTest = evaluateResults(bestModel, hashTestData)

# Log loss for the baseline model
logLossTestBaseline = hashTestData.map(lambda lp: computeLogLoss(classOneFracTrain, lp.label)).mean()

print ('Hashed Features Test Log Loss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
      .format(logLossTestBaseline, logLossTest))

# TEST Evaluate on the test set (5e)
 
print ('OHE Features Validation Logloss:\n\tBaseline = {0:.3f}\n\tLogReg = {1:.3f}'
       .format(logLossValBase, logLossValLR0))

Test.assertTrue(np.allclose(logLossTestBaseline, 0.537438),
                'incorrect value for logLossTestBaseline')
Test.assertTrue(np.allclose(logLossTest, 0.455616931), 'incorrect value for logLossTest')

