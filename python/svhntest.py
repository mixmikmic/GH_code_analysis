import numpy as np
import matplotlib.pyplot as plt
import svhn
import graphics
import keras_utils
from keras.utils import np_utils
import keras
from keras import backend as K

get_ipython().magic('matplotlib inline')

max_digits = 7
image_size = (54,128)

# print the keras version used
import keras
print "Keras version : {}".format(keras.__version__)

# load the data file (takes time)
rawdata = svhn.read_process_h5('../inputs/test/digitStruct.mat')

# extract resized images, counts and label sequences for each sample
def generateTestData(data, n=100):
    Ximg = []
    ycount = []
    ylabel = []
    for datapoint in np.random.choice(data, size=n, replace=False):
        img,rawsize = svhn.createImageData(datapoint, image_size, '../inputs/test/')
        Ximg.append(img)
        ycount.append(datapoint['length'])
        ylabel.append(datapoint['labels'])
        
    ylabel = [[0 if y==10 else int(y) for y in ys] for ys in ylabel]
    return np.array(Ximg), np.array(ycount), np.array(ylabel)

def standardize(img):
    s = img - np.mean(img, axis=(2,0,1), keepdims=True)
    s /= np.std(s, axis=(2,0,1), keepdims=True)
    return s
    
# change to 13068 to test on full test dataset
Ximg, ycount, ylabel = generateTestData(rawdata, 13068)
Xs = np.array([standardize(x) for x in Ximg])

model_yaml = open('../checkpoints/model.yaml','r')
model = keras.models.model_from_yaml(model_yaml.read())
model_yaml.close()
model.load_weights('../checkpoints/model.hdf5')

# enumerate the layers of main graph
for i,layer in zip(range(len(model.layers)), model.layers):
    print "layer {} : {}".format(i,layer.name)

# extract the individual models from training graph
vision = model.layers[1]
counter = model.layers[3]
detector = model.layers[4]

h = vision.predict(Xs)

ycount_ = counter.predict(h)
ycount_ = np.argmax(ycount_, axis=1)

ylabel_ = []
for i in range(len(ycount_)):
    # generate range for each count
    indices = np.arange(ycount_[i])
    # one hot encoding for each index
    indices = np_utils.to_categorical(indices, max_digits)
    # tile h to match shape of indices matrix
    hs = np.tile(h[i], (ycount_[i],1))
    
    # predict labels for the sample
    sample_seq = detector.predict([hs, indices])
    sample_seq = np.argmax(sample_seq,1)
    ylabel_.append(sample_seq)

from sklearn.metrics import classification_report
print classification_report(ycount, ycount_)

ycmin = np.minimum(ycount, ycount_)

# extract labels from ylabel and ylabel_ using ycmin
ylabel_det = np.array([ylabelrow[0:ycminc] for ylabelrow,ycminc in zip(ylabel, ycmin)])
ylabel_det = np.concatenate(ylabel_det)

ylabel_det_= np.array([ylabelrow[0:ycminc] for ylabelrow,ycminc in zip(ylabel_, ycmin)])
ylabel_det_= np.concatenate(ylabel_det_)

print classification_report(ylabel_det, ylabel_det_)

def matchSequence(seq, seq_):
    return [np.array_equal(seqi, seqi_) for seqi, seqi_ in zip(seq, seq_)]
seqmatch = matchSequence(ylabel, ylabel_)
print "Sequence prediction accuracy : {}".format(np.average(seqmatch))

graphics.displaySamples(Ximg, ycounttrue=ycount, ycountpred=ycount_, ylabels=ylabel, ylabelspred=ylabel_)

convlayer = 2
graphics.showCNNConv(vision, convlayer, Ximg[4], standardize(Ximg[4]))

convlayer = 4
graphics.showCNNConv(vision, convlayer, Ximg[4], standardize(Ximg[4]))



