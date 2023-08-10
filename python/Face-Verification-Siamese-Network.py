get_ipython().magic('matplotlib inline')
# Imports
import numpy as np
from itertools import combinations
import random

from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

# For keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from keras.initializers import Identity, RandomNormal
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator

# TensorFlow initialization
import tensorflow as tf
session = tf.Session()
from keras import backend as K
K.set_session(session)

def createKochTwin(inputDim, normalStddev, biasMean, lambdaConv, lambdaDense):
	'''
	Builds a Siamese twin to be shared by the two inputs. Architecture used in Koch et al. (2015).

	@param: inputDim - shape of the input image
	@param: normalStddev - standard deviation of the zero mean Gaussian to be used for the weight initialization
	@param: biasMean - mean of the Gaussian to be used for bias initialization
	@param: lambdaConv - regularization parameter for the convolutional layers
	@param: lambdaDense - regularization parameter for the dense layers
	'''
	twin = Sequential()
	twin.add(Conv2D(64, (10, 10), activation='relu', input_shape=inputDim, kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv)))
	twin.add(MaxPooling2D()) # default pool size pool_size=(2, 2)

	twin.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	twin.add(MaxPooling2D())

	twin.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	twin.add(MaxPooling2D())

	twin.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	twin.add(Flatten())
	twin.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(lambdaDense), kernel_initializer=RandomNormal(stddev=normalStddev), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	return twin

def createCompressedTwin(inputDim, normalStddev, biasMean, lambdaConv, lambdaDense):
	'''
	Builds a Siamese twin to be shared by the two inputs. A trimmed network architecture based on Koch et al. (2015).

	@param: inputDim - shape of the input image
	@param: normalStddev - standard deviation of the zero mean Gaussian to be used for the weight initialization
	@param: biasMean - mean of the Gaussian to be used for bias initialization
	@param: lambdaConv - regularization parameter for the convolutional layers
	@param: lambdaDense - regularization parameter for the dense layers
	'''
	twin = Sequential()
	twin.add(Conv2D(16, (6, 6), activation='relu', input_shape=inputDim, kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv)))
	twin.add(MaxPooling2D()) # default pool size pool_size=(2, 2)

	twin.add(Conv2D(32, (4, 4), activation='relu', kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	twin.add(MaxPooling2D())

	twin.add(Conv2D(32, (4, 4), activation='relu', kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	
	twin.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer=RandomNormal(stddev=normalStddev), kernel_regularizer=l2(lambdaConv), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	twin.add(Flatten())
	twin.add(Dense(256, activation='sigmoid', kernel_regularizer=l2(lambdaDense), kernel_initializer=RandomNormal(stddev=normalStddev), bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev)))
	return twin

def l1Distance(vects):
	'''
	Finds the component wise l1 distance to join the twins. To be used in the Lambda layer definition.

	@param: vects - pair of images processed through the Siamese network

	@return: absolute value of the difference between the two element in vects
	'''
	return K.abs(vects[0] - vects[1])

def l1OutputShape(shapes):
	'''
	Returns the shape of the resulting after the twins are joined. To be used in the Lambda layer definition.

	@param: shapes - shapes of the pair of images processed through the Siamese network

	@return: shapes of the first processed image
	'''
	return shapes[0]

# Loading the dataset
loadedData = fetch_olivetti_faces()
numSamplesPerClass = 10
numTrainClasses = 38
uniqueClasses = list(set(loadedData.target))

IMG_SIZE = 64 # Using the actual size
inputDim = (IMG_SIZE, IMG_SIZE, 1)
allImages = loadedData.images 
# Reshaping the data for ease of indexing
allImagesByClass = np.reshape(allImages, (len(uniqueClasses), numSamplesPerClass, IMG_SIZE, IMG_SIZE, 1))

data_gen_args = dict(
	rotation_range=10,
	width_shift_range=0.05,
	height_shift_range=0.05,
	horizontal_flip=True)

image_datagen = ImageDataGenerator(**data_gen_args)

maxDraws = 6 # Number of transformations per image
augmentedImagesByClass = [[] for k in range(40)]
for nClass in uniqueClasses:
	numDraws = 0
	emptArr = []
	for _x, _y in image_datagen.flow(allImagesByClass[nClass], [nClass]*numSamplesPerClass):
		emptArr += _x
		numDraws += 1
		if numDraws == 1:
			augmentedImagesByClass[nClass] = _x
		elif numDraws <= maxDraws:
			augmentedImagesByClass[nClass] = np.concatenate((augmentedImagesByClass[nClass], _x))
		if numDraws >= maxDraws:
			break
augmentedImagesByClass = np.array(augmentedImagesByClass)

numFolds = len(uniqueClasses)
# Lists to store the results of the cross validation folds
valScoresList = [[] for k in range(numFolds)]
test1ScoresList = [[] for k in range(numFolds)]
test2ScoresList = [[] for k in range(numFolds)]

# Hyperparmaters used for the network
lambdaConv = 1e-4
lambdaDense = 1e-4
normalStddev = 1e-2
biasMean = 0.5
numEpochs = 10
miniBatchSize = 128

# Cross validation loop
for testClass in range(numFolds):
	trainClasses = uniqueClasses[:testClass]+uniqueClasses[testClass+1:]
	randClass = random.randrange(1, numTrainClasses) # Pick a random class for validation set
	valClass = trainClasses.pop(randClass)

	# Creating training set
	# Training data has equal distribution of same and different class images
	trainPairs = []
	trainLabels = []
	for nClass in trainClasses:
		for nSample1, nSample2 in combinations(range(numSamplesPerClass*6),2):
			trainPairs += [[augmentedImagesByClass[nClass, nSample1], augmentedImagesByClass[nClass, nSample2]]]
			inc = random.randrange(1, numTrainClasses)
			idx = (nClass + inc) % numTrainClasses
			nClass2 = trainClasses[idx]
			trainPairs += [[augmentedImagesByClass[nClass, nSample1], augmentedImagesByClass[nClass2, nSample2]]]
			trainLabels += [1, 0]
	trainPairs = np.array(trainPairs)
	trainLabels = np.array(trainLabels)

	# Creating validation set
	# Validation data has equal distribution of same and different class images
	valPairs = []
	valLabels = []
	for nSample1, nSample2 in combinations(range(numSamplesPerClass),2):
		valPairs += [[allImagesByClass[valClass, nSample1], allImagesByClass[valClass, nSample2]]]
		inc = random.randrange(1, numTrainClasses)
		nClass2 = trainClasses[inc % numTrainClasses]
		valPairs += [[allImagesByClass[valClass, nSample1], allImagesByClass[nClass2, nSample2]]]
		valLabels += [1, 0]
	valPairs = np.array(valPairs)
	valLabels = np.array(valLabels)

	# Creating test set 1
	# Constructed by drawing a pair of images from held out test class
	testPairs = []
	testLabels = []
	for nSample1, nSample2 in combinations(range(numSamplesPerClass),2):
		testPairs += [[allImagesByClass[testClass, nSample1], allImagesByClass[testClass, nSample2]]]
		testLabels += [1]
	testPairs = np.array(testPairs)
	testLabels = np.array(testLabels)

	# Creating test set 2
	# Constructed by drawing one image from the held out test class and another from any other class
	testPairs2 = []
	testLabels2 = []
	for nClass in trainClasses:
		for nSample1 in range(numSamplesPerClass):
			for nSample2 in range(numSamplesPerClass):
				testPairs2 += [[allImagesByClass[testClass, nSample1], allImagesByClass[nClass, nSample2]]]
				testLabels2 += [0]
	testPairs2 = np.array(testPairs2)
	testLabels2 = np.array(testLabels2)

	# Network model construction
	baseTwin = createCompressedTwin(inputDim, normalStddev, biasMean, lambdaConv, lambdaDense)
	inputL = Input(inputDim)
	inputR = Input(inputDim)
	processedL = baseTwin(inputL)
	processedR = baseTwin(inputR)

	featureVector = Lambda(l1Distance, output_shape=l1OutputShape)([processedL, processedR])
	sigmoidOutput = Dense(1, activation='sigmoid', bias_initializer=RandomNormal(mean=biasMean, stddev=normalStddev))(featureVector)
	siameseModel = Model(input=[inputL, inputR], output=sigmoidOutput)
	siameseModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Training loop
	for k in range(numEpochs):
		print 'Epoch', k
		siameseModel.fit([trainPairs[:, 0], trainPairs[:, 1]], trainLabels, batch_size=128, epochs=1, verbose=0)
		
		valScores = siameseModel.evaluate([valPairs[:, 0], valPairs[:, 1]], valLabels, verbose=0)
		valScoresList[testClass].append(valScores[1])

		test1Scores = siameseModel.evaluate([testPairs[:, 0], testPairs[:, 1]], testLabels, verbose=0)
		test1ScoresList[testClass].append(test1Scores[1])

		test2Scores = siameseModel.evaluate([testPairs2[:, 0], testPairs2[:, 1]], testLabels2, verbose=0)
		test2ScoresList[testClass].append(test2Scores[1])

	# Caching prediction at the end of training to get RoC/DET curve 
	positiveSamplesScores = siameseModel.predict([testPairs[:, 0], testPairs[:, 1]])
	negativeSamplesScores = siameseModel.predict([testPairsInTrain[:, 0], testPairsInTrain[:, 1]])

	siamesePrediction = np.concatenate((positiveSamplesScores,negativeSamplesScores))
	trueLabel = np.concatenate((testLabels, testLabelsInTrain))
	fpr, tpr, thresholds = roc_curve(trueLabel, siamesePrediction)

accurayByClass = [None for k in range(numFolds)]
for testClass in range(numFolds):
	epochStop = valScoresList.index(max(valScoresList[testClass]))
	accurayByClass[testClass] = (test1ScoresList[testClass][epochStop]+test2ScoresList[testClass][epochStop])/2.0

lineNumber = 0
for line in open('rocResults.txt'):
	content = line.strip().split(',')
	if lineNumber%4==0:
		fpr = [float(elem)*100 for elem in content]
	elif lineNumber%4==1:
		mdr = [100.0-float(elem)*100 for elem in content]
	elif lineNumber%4==2:
		threshold = [float(elem) for elem in content]
	elif lineNumber%4==3:
		plt.plot(fpr, mdr)
	lineNumber = lineNumber + 1
plt.ylabel('Missed Detection Rate (%)')
plt.xlabel('False Positive Rate (%)')
plt.title('Detection Error Tradeoff Curve')
plt.ylim([0.0,20.0])
plt.xlim([0.0,20.0])

