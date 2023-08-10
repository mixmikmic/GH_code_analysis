get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import warnings
import random
from datetime import datetime
random.seed(datetime.now())
warnings.filterwarnings('ignore')

# Make plots larger
plt.rcParams['figure.figsize'] = (10, 6)

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
import cv2 # pip install opencv-python
import sys
import os
from random import shuffle, randint, choice

# Generates random image with squares and circles
def getRandomImage():
	imageSize = 100
	size = 25
	nbShapes = 5

	xy = lambda: randint(0,100)

	# Create a white image
	img = np.zeros((imageSize,imageSize,3), np.uint8)
	cv2.rectangle(img,(0,0),(imageSize,imageSize),(122,122,122) ,-1)

	greyImg = np.copy(img)

	# Adds some shapes
	for i in range(nbShapes):
		x0, y0 = xy(), xy()
		isRect = choice((True,False))
		if isRect:
			cv2.rectangle(img,(x0,y0),(x0+size,y0+size),(255,0,0) ,-1)
			cv2.rectangle(greyImg,(x0,y0),(x0+size,y0+size),(255,255,255) ,-1)
		else:
			cv2.circle(img,(x0,y0), size/2, (0,0,255), -1)
			cv2.circle(greyImg,(x0,y0), size/2, (255,255,255), -1)

	return cv2.resize(img,(48,48)), cv2.resize(greyImg,(48,48))

# Creates the dataset
def getDataset(display=False):
	# Show what the dataset looks like
	if display:
		colorImg, greyImg = getRandomImage()
		img = np.hstack((colorImg, greyImg))
		cv2.imshow("Dataset",cv2.resize(img,(200,100)))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	#for i in range

	x_train, x_test, y_train, y_test = [], [], [], []

	# Add training examples
	for i in range(10000):
		colorImg, greyImg = getRandomImage()
		greyImg = cv2.cvtColor(greyImg, cv2.COLOR_RGB2GRAY)
		x_train.append(greyImg.astype('float32')/255.)
		y_train.append(colorImg.astype('float32')/255.)

	# Add test examples
	for i in range(1000):
		colorImg, greyImg = getRandomImage()
		greyImg = cv2.cvtColor(greyImg, cv2.COLOR_RGB2GRAY)
		x_test.append(greyImg.astype('float32')/255.)
		y_test.append(colorImg.astype('float32')/255.)

	# Reshape
	x_train = np.array(x_train).reshape((-1,48,48,1))
	x_test = np.array(x_test).reshape((-1,48,48,1))
	y_train = np.array(y_train).reshape((-1,48,48,3))
	y_test = np.array(y_test).reshape((-1,48,48,3))

	return x_train, y_train, x_test, y_test

# Creates the Convolutional Auto Encoder
def getModel():
	input_img = Input(shape=(48, 48, 1))
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
	#6x6x32 -- bottleneck
	x = UpSampling2D((2, 2), dim_ordering='tf')(encoded)
	x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	x = UpSampling2D((2, 2), dim_ordering='tf')(x)
	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
	decoded = Convolution2D(3, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)

	#Create model
	autoencoder = Model(input_img, decoded)
	return autoencoder

# Trains the model for 10 epochs
def trainModel():
	# Load dataset
	print("Loading dataset...")
	x_train_gray, x_train, x_test_gray, x_test = getDataset()

	# Create model description
	print("Creating model...")
	model = getModel()
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

	# Train model
	print("Training model...")
	model.fit(x_train_gray, x_train, nb_epoch=10, batch_size=148, shuffle=True, validation_data=(x_test_gray, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

	# Evaluate loaded model on test data
	print("Evaluating model...")
	score = model.evaluate(x_train_gray, x_train, verbose=0)
	print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

	# Serialize model to JSON
	print("Saving model...")
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	    
	# Serialize weights to HDF5
	print("Saving weights...")
	model.save_weights("model.h5")

# Tests the model and shows results
def testModel():
	# Load JSON model description
	with open('model.json', 'r') as json_file:
		modelJSON = json_file.read()
	
	# Build model from JSON description
	print("Loading model...")
	model = model_from_json(modelJSON)

	# Load weights
	print("Loading weights...")
	model.load_weights("model.h5")

	_, _, x_test_gray, x_test = getDataset()
	x_test_gray = x_test_gray[:10]
	x_test = x_test[:10]

	print("Making predictions...")
	predictions = model.predict(x_test_gray)
	x_test_gray = [cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) for img in x_test_gray]

	img = np.vstack((np.hstack(x_test_gray),np.hstack(predictions),np.hstack(x_test)))

	cv2.imshow("Input - Reconstructed - Ground truth",cv2.resize(img,(img.shape[1],img.shape[0])))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot first 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

import os
import sys
import h5py
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import norm
from sklearn import manifold

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras import objectives

# Config

modelsPath = "Models/"
visualsPath = "Visuals/"

imageSize = 28
latent_dim = 32

# Convolutional models
# x is input, z is 
def getModels():
    input_img = Input(shape=(imageSize, imageSize, 1))
    x = Convolution2D(32, 3, 3, border_mode='same')(input_img)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    # Latent space // bottleneck layer
    x = Flatten()(x)
    x = Dense(latent_dim)(x)
    z = ELU()(x)

    ##### MODEL 1: ENCODER #####
    encoder = Model(input_img, z)
   
    # We instantiate these layers separately so as to reuse them for the decoder
    # Dense from latent space to image dimension
    x_decoded_dense1 = Dense(7 * 7 * 64)

    # Reshape for image
    x_decoded_reshape0 = Reshape((7, 7, 64))
    x_decoded_upsample0 = UpSampling2D((2, 2))
    x_decoded_conv0  = Convolution2D(32, 3, 3, border_mode='same')

    x_decoded_upsample3 = UpSampling2D((2, 2))
    x_decoded_conv3 = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')

    # Create second part of autoencoder
    x_decoded = x_decoded_dense1(z)
    x_decoded = ELU()(x_decoded)

    x_decoded = x_decoded_reshape0(x_decoded)
    x_decoded = x_decoded_upsample0(x_decoded)
    x_decoded = x_decoded_conv0(x_decoded)
    x_decoded = ELU()(x_decoded)

    # Tanh layer
    x_decoded = x_decoded_upsample3(x_decoded)
    decoded_img = x_decoded_conv3(x_decoded)

    ##### MODEL 2: AUTO-ENCODER #####
    autoencoder = Model(input_img, decoded_img)

    # Create decoder
    input_z = Input(shape=(latent_dim,))
    x_decoded_decoder = x_decoded_dense1(input_z)
    x_decoded_decoder = ELU()(x_decoded_decoder)

    x_decoded_decoder = x_decoded_reshape0(x_decoded_decoder)
    x_decoded_decoder = x_decoded_upsample0(x_decoded_decoder)
    x_decoded_decoder = x_decoded_conv0(x_decoded_decoder)
    x_decoded_decoder = ELU()(x_decoded_decoder)

    # Tanh layer
    x_decoded_decoder = x_decoded_upsample3(x_decoded_decoder)
    decoded_decoder_img = x_decoded_conv3(x_decoded_decoder)

    ##### MODEL 3: DECODER #####
    decoder = Model(input_z, decoded_decoder_img)
    return autoencoder, encoder, decoder

import os
import sys
import h5py
import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import norm
from sklearn import manifold

from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Convolution2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras.layers.advanced_activations import ELU
from keras import backend as K
from keras import objectives

# Convolutional models
# x is input, z is 
def getModels():
    input_img = Input(shape=(imageSize, imageSize, 1))
    x = Convolution2D(32, 3, 3, border_mode='same')(input_img)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = ELU()(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)

    # Latent space // bottleneck layer
    x = Flatten()(x)
    x = Dense(latent_dim)(x)
    z = ELU()(x)

    ##### MODEL 1: ENCODER #####
    encoder = Model(input_img, z)
   
    # We instantiate these layers separately so as to reuse them for the decoder
    # Dense from latent space to image dimension
    x_decoded_dense1 = Dense(7 * 7 * 64)

    # Reshape for image
    x_decoded_reshape0 = Reshape((7, 7, 64))
    x_decoded_upsample0 = UpSampling2D((2, 2))
    x_decoded_conv0  = Convolution2D(32, 3, 3, border_mode='same')

    x_decoded_upsample3 = UpSampling2D((2, 2))
    x_decoded_conv3 = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')

    # Create second part of autoencoder
    x_decoded = x_decoded_dense1(z)
    x_decoded = ELU()(x_decoded)

    x_decoded = x_decoded_reshape0(x_decoded)
    x_decoded = x_decoded_upsample0(x_decoded)
    x_decoded = x_decoded_conv0(x_decoded)
    x_decoded = ELU()(x_decoded)

    # Tanh layer
    x_decoded = x_decoded_upsample3(x_decoded)
    decoded_img = x_decoded_conv3(x_decoded)

    ##### MODEL 2: AUTO-ENCODER #####
    autoencoder = Model(input_img, decoded_img)

    # Create decoder
    input_z = Input(shape=(latent_dim,))
    x_decoded_decoder = x_decoded_dense1(input_z)
    x_decoded_decoder = ELU()(x_decoded_decoder)

    x_decoded_decoder = x_decoded_reshape0(x_decoded_decoder)
    x_decoded_decoder = x_decoded_upsample0(x_decoded_decoder)
    x_decoded_decoder = x_decoded_conv0(x_decoded_decoder)
    x_decoded_decoder = ELU()(x_decoded_decoder)

    # Tanh layer
    x_decoded_decoder = x_decoded_upsample3(x_decoded_decoder)
    decoded_decoder_img = x_decoded_conv3(x_decoded_decoder)

    ##### MODEL 3: DECODER #####
    decoder = Model(input_z, decoded_decoder_img)
    return autoencoder, encoder, decoder

# Loads the dataset - here Normalized MNIST
def loadDataset():
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape([-1,28,28,1])/255.
    X_test = X_test.reshape([-1,28,28,1])/255.
    return X_train, X_test

import os
import sys
import h5py
import cv2
import math
import random, string

import numpy as np
from scipy.stats import norm
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Show every image, good for picking interplation candidates
def visualizeDataset(X):
    for i,image in enumerate(X):
        cv2.imshow(str(i),image)
        cv2.waitKey()
        cv2.destroyAllWindows()

# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([imageSize,imageSize])
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()

# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(X, encoder, display=True):
    # Compute latent space representation
    print("Computing latent space projection...")
    X_encoded = encoder.predict(X)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne

# Show dataset images with T-sne projection of pixel space
def computeTSNEProjectionOfPixelSpace(X, display=True):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1,imageSize*imageSize*1]))

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne

# Reconstructions for samples in dataset
def getReconstructedImages(X, autoencoder):
    nbSamples = X.shape[0]
    nbSquares = int(math.sqrt(nbSamples))
    nbSquaresHeight = 2*nbSquares
    nbSquaresWidth = nbSquaresHeight
    resultImage = np.zeros((nbSquaresHeight*imageSize,nbSquaresWidth*imageSize/2,X.shape[-1]))

    reconstructedX = autoencoder.predict(X)

    for i in range(nbSamples):
        original = X[i]
        reconstruction = reconstructedX[i]
        rowIndex = i%nbSquaresWidth
        columnIndex = (i-rowIndex)/nbSquaresHeight
        resultImage[rowIndex*imageSize:(rowIndex+1)*imageSize,columnIndex*2*imageSize:(columnIndex+1)*2*imageSize,:] = np.hstack([original,reconstruction])

    return resultImage

# Reconstructions for samples in dataset
def visualizeReconstructedImages(X_train, X_test, autoencoder, save=False, label=False):
    trainReconstruction = getReconstructedImages(X_train,autoencoder)
    testReconstruction = getReconstructedImages(X_test,autoencoder)

    if not save:
        print("Generating 10 image reconstructions...")

    result = np.hstack([trainReconstruction,np.zeros([trainReconstruction.shape[0],5,trainReconstruction.shape[-1]]),testReconstruction])
    result = (result*255.).astype(np.uint8)

    if save:
        cv2.imwrite(visualsPath+"reconstructions_{}.png".format(label),result)
    else:
        cv2.imshow("Reconstructed images (train - test)",result)
        cv2.waitKey()
        cv2.destroyAllWindows()

# Computes A, B, C, A+B, A+B-C in latent space
def visualizeArithmetics(a, b, c, encoder, decoder):
    print("Computing arithmetics...")
    # Create micro batch
    X = np.array([a,b,c])

    # Compute latent space projection
    latentA, latentB, latentC = encoder.predict(X)

    add = latentA+latentB
    addSub = latentA+latentB-latentC

    # Create micro batch
    X = np.array([latentA,latentB,latentC,add,addSub])

    # Compute reconstruction
    reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub = decoder.predict(X)

    cv2.imshow("Arithmetics in latent space",np.hstack([reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub]))
    cv2.waitKey()

# Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(start, end, encoder, decoder, save=False, nbSteps=5):
    print("Generating interpolations...")

    # Create micro batch
    X = np.array([start,end])

    # Compute latent space projection
    latentX = encoder.predict(X)
    latentStart, latentEnd = latentX

    # Get original image for comparison
    startImage, endImage = X

    vectors = []
    normalImages = []
    #Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart*(1-alpha) + latentEnd*alpha
        vectors.append(vector)
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage,1-alpha,endImage,alpha,0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    reconstructions = decoder.predict(vectors)

    # Put final image together
    resultLatent = None
    resultImage = None

    if save:
        hashName = ''.join(random.choice(string.lowercase) for i in range(3))

    for i in range(len(reconstructions)):
        interpolatedImage = normalImages[i]*255
        interpolatedImage = cv2.resize(interpolatedImage,(50,50))
        interpolatedImage = interpolatedImage.astype(np.uint8)
        resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage,interpolatedImage])

        reconstructedImage = reconstructions[i]*255.
        reconstructedImage = reconstructedImage.reshape([28,28])
        reconstructedImage = cv2.resize(reconstructedImage,(50,50))
        reconstructedImage = reconstructedImage.astype(np.uint8)
        resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent,reconstructedImage])
    
        if save:
            cv2.imwrite(visualsPath+"{}_{}.png".format(hashName,i),np.hstack([interpolatedImage,reconstructedImage]))

        result = np.vstack([resultImage,resultLatent])

    if not save:
        cv2.imshow("Interpolation in Image Space vs Latent Space",result)
        cv2.waitKey()
        cv2.destroyAllWindows()

