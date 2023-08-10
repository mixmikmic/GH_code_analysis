import os
import numpy as np
import theano
import theano.tensor as T
import lasagne

# defining the VGG-16 model
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

import cPickle
d = cPickle.load(open('vgg16_2.pkl'))

net = build_model()

lasagne.layers.set_all_param_values(net['prob'], d['param values'])

import cv2
IMAGE_MEAN = d['mean value'][:, np.newaxis, np.newaxis]

# this will be the function used to resize and grayscale the the raw input image
def resize(img):
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    return img

def perCentreMean(img):
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    img = img - IMAGE_MEAN
    return img

# for efficient memory storage :
# since the numbers are between 0 and 255, numpy's int16 datatype can be used
# secondly, we need to flatten the image array to store it efficiently ad row-major
# so for that, we will numpy's use numpy's ravel function with 'copy' flag = False
def generateDataForImages():
    """
    In this function, various techniques for augementing image data, such as random rotations
    and translations, zooming on images will performed to produce a dataset of size ~1600 images.
    RETURN :
        a list of 2-tuples where the first element in each tuple is an image's numpy 2-D array 
        and the second label is the corresponding label, 1 for real note and 0 for fake note
    """
    
    # a list of 2-tuples to hold an image's numpy 2-D array 
    # and the corresponding label, 1 for real and 0 for fake
    trainingInput = []
    trainingOutput = []
    noOfRows = 224
    noOfCols = 224
    
    # first all real currency notes' images and then all the fake ones'
    # i am skeptical of the consequences :-/
    label = 1
    inputArray = False
    
    for directory in ['Real Notes 2000/', 'Duplicate Notes 2000/']:
        for filename in os.listdir(directory):
            print filename
            img = cv2.imread(directory + filename)
            img = resize(img)
            
            # TRANSLATIONS
            # will produce (6 + 6 + 1) x (6 + 6 + 1) = 169 images
            # stride of 5 pixels along both axis along all 4 directions
            for x in range(30, -35, -5):
                for y in range(30, -35, -5):
                    translationMatrix = np.float32([ [1,0,x], [0,1,y] ])
                    imgTrns = cv2.warpAffine(img, translationMatrix, (noOfCols, noOfRows))
                    imgTrns = perCentreMean(imgTrns)
                    trainingInput.append(floatX(imgTrns[np.newaxis]))
                    trainingOutput.append(label)

            # ROTATIONS
            # we produce 41 different angles in the range of -10 to 10
            # with the step being equal to 0.5
            for angle in range(20, -21, -1):
                rotationMatrix = cv2.getRotationMatrix2D((noOfCols/2, noOfRows/2), float(angle)/2, 1)
                imgRotated = cv2.warpAffine(img, rotationMatrix, (noOfCols, noOfRows))
                imgRotated = perCentreMean(imgRotated)
                trainingInput.append(floatX(imgRotated[np.newaxis]))
                trainingOutput.append(label)

            # PROJECTIVE TRANSFORMATIONS for ZOOMING IN AND ZOOMING OUT
            # will produce (30 + 30) images for the dataset
            # 1ST ZOOMING IN ...
            for step in np.arange(0.001, 0.031, 0.001):
                srcPoints = np.float32([[int(step*(noOfCols-1)),int(step*(noOfRows-1))], [int((1-step)*(noOfCols-1)),int(step*(noOfRows-1))], [int(step*(noOfCols-1)),int((1-step)*(noOfRows-1))], [int((1-step)*(noOfCols-1)), int((1-step)*(noOfRows-1))]])
                dstPoints = np.float32([[0,0], [noOfCols-1,0], [0,noOfRows-1], [noOfCols-1,noOfRows-1]]) 
                projective_matrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
                imgZoomed = cv2.warpPerspective(img, projective_matrix, (noOfCols,noOfRows))
                imgZoomed = perCentreMean(imgZoomed)
                trainingInput.append(floatX(imgZoomed[np.newaxis]))
                trainingOutput.append(label)
            # 2ND ZOOMING OUT ...
            for step in np.arange(0.001, 0.031, 0.001):
                srcPoints = np.float32(np.float32([[0,0], [noOfCols-1,0], [0,noOfRows-1], [noOfCols-1,noOfRows-1]]))
                dstPoints = np.float32([[int(step*(noOfCols-1)),int(step*(noOfRows-1))], [int((1-step)*(noOfCols-1)),int(step*(noOfRows-1))], [int(step*(noOfCols-1)),int((1-step)*(noOfRows-1))], [int((1-step)*(noOfCols-1)), int((1-step)*(noOfRows-1))]]) 
                projective_matrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
                imgZoomed = cv2.warpPerspective(img, projective_matrix, (noOfCols,noOfRows))
                imgZoomed = perCentreMean(imgZoomed)
                trainingInput.append(floatX(imgZoomed[np.newaxis]))
                trainingOutput.append(label)

        # set label for fake images to come
        label = 0
       
    return trainingInput, trainingOutput

X, y = generateDataForImages()

X = np.concatenate(X)
y = np.array(y).astype('int32')

import sklearn.model_selection
X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

del d

del X
del y

output_layer = DenseLayer(net['fc7'], num_units=2, nonlinearity=softmax)

X_sym = T.tensor4()
y_sym = T.ivector()

prediction = lasagne.layers.get_output(output_layer, X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                      dtype=theano.config.floatX)

params = lasagne.layers.get_all_params(output_layer, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.0001, momentum=0.9)

train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)

BATCH_SIZE = 1
for epoch in range(5):
    for i in xrange(len(X_tr)):
        loss = train_fn(X_tr[i: i+BATCH_SIZE], y_tr[i: i+BATCH_SIZE])
        print 'iteration {0} complete'.format(i)
    print "Epoch {0} complete.".format(epoch)

model = lasagne.layers.get_all_param_values(net['fc7'])

with open('vgg16FakeCurrencyDetectionWeights.pkl','wb') as f:
    cPickle.dump(model, f, protocol = 2)

shared_x = theano.shared(
        np.asarray(X_val, dtype=theano.config.floatX), borrow=True)
shared_y = theano.shared(
        np.asarray(y_val, dtype=theano.config.floatX), borrow=True)

cummLoss = 0
cummAcc = 0
for i in xrange(len(X_val)):
    loss, acc = val_fn(X_val[i:i+BATCH_SIZE], y_val[i:i+BATCH_SIZE])
    cummLoss += loss
    cummAcc += acc
    print 'Image {0} classified'.format(i+1)

loss = cummLoss/len(X_val)
acc = cummAcc/len(y_val)
print 'Accuracy = ', acc

loss

