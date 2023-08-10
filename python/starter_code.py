import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color, img_as_bool, exposure, transform
get_ipython().magic('matplotlib inline')

# location of original images
subdirectory = 'images/'

# read csv with pixel locations of aircraft and label updates
new_labels_x = pd.read_csv('new_plane_labels.csv')
print(new_labels_x.head())
print(new_labels_x.dtypes)
print(new_labels_x.shape)

# look at the labels being disregarded
bad_labels = new_labels_x[new_labels_x['good_label']==False]
print(bad_labels.shape)
for index, row in bad_labels.iterrows():
    toRead = subdirectory + row['img_name']
    img_raw = io.imread(toRead)
    plt.figure()
    io.imshow(img_raw)

# filter to include only the *good* labels for training
new_labels = new_labels_x[new_labels_x['good_label']==True]
print(new_labels.shape)

features_list = []
y_list = []
imnames_list = []

# get and look at examples of images containing aircraft
import warnings
warnings.filterwarnings('ignore')

crop_pixels = 20 # number of pixels by which the crop will be furthered

for index, row in new_labels.iterrows():
    toRead = subdirectory + row['img_name']
    img_raw = io.imread(toRead)
    img_cropped = img_raw[row['y1_pixel'] + crop_pixels : row['y1_pixel'] + 90 - crop_pixels, 
                          row['x1_pixel'] + crop_pixels : row['x1_pixel'] + 160 - crop_pixels]
    img_rs = transform.rescale(img_cropped, 0.8)
    img_gray = color.rgb2gray(img_rs)
    p1, p2 = np.percentile(img_gray, (3, 97))
    img_rescale = exposure.rescale_intensity(img_gray, in_range=(p1, p2))
    img_bool = img_as_bool(img_rescale)
    final_image = img_bool
    # save the final image to features_list
    features_list.append(final_image)
    imnames_list.append(row['img_name'])
    y_list.append(True)
    # view the image
    plt.figure()
    io.imshow(final_image)

# read labels for aircraft images
labels = pd.read_csv('aircraft.csv')
print(labels.head())
print(labels.shape)

# create list of images that do not contain aircraft
no_aircraft = labels[labels['aircraft']==False]['imageName']
print(no_aircraft.shape)
print(type(no_aircraft))

# features for non-aircraft
from random import randrange, seed

seed(5)
i = 0

for notplane in no_aircraft:
    toRead = subdirectory + notplane
    img_raw = io.imread(toRead)
    # select a random area to begin the crop to 160x90
    y1 = randrange(360-90)
    x1 = randrange(640-160)
    img_cropped = img_raw[y1 + crop_pixels : y1 + 90 - crop_pixels, x1 + crop_pixels : x1 + 160 - crop_pixels]
    img_rs = transform.rescale(img_cropped, 0.8)
    img_gray = color.rgb2gray(img_rs)
    p2, p98 = np.percentile(img_gray, (3, 97))
    img_rescale = exposure.rescale_intensity(img_gray, in_range=(p2, p98))
    img_bool = img_as_bool(img_rescale)
    features_list.append(img_bool)
    imnames_list.append(notplane)
    y_list.append(False)
    i = i + 1
    if i < 50:
        plt.figure()
        io.imshow(img_bool)

class BinaryClassificationPerformance():
    '''Performance measures to evaluate the fit of a binary classification model'''
    
    def __init__(self, predictions, labels, desc, probabilities=None):
        '''Initialize attributes: predictions-vector of predicted values for Y, labels-vector of labels for Y'''
        '''probabilities-optional, probability that Y is equal to True'''
        self.probabilities = probabilities
        self.performance_df = pd.concat([pd.DataFrame(predictions), pd.DataFrame(labels)], axis=1)
        self.performance_df.columns = ['preds', 'labls']
        self.desc = desc
        self.performance_measures = {}
        self.image_indices = {}
  
    def compute_measures(self):
        '''Compute performance measures defined by Flach p. 57'''
        self.performance_measures['Pos'] = self.performance_df['labls'].sum()
        self.performance_measures['Neg'] = self.performance_df.shape[0] - self.performance_df['labls'].sum()
        self.performance_measures['TP'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == True)).sum()
        self.performance_measures['TN'] = ((self.performance_df['preds'] == False) & (self.performance_df['labls'] == False)).sum()
        self.performance_measures['FP'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == False)).sum()
        self.performance_measures['FN'] = ((self.performance_df['preds'] == False) & (self.performance_df['labls'] == True)).sum()
        self.performance_measures['Accuracy'] = (self.performance_measures['TP'] + self.performance_measures['TN']) / (self.performance_measures['Pos'] + self.performance_measures['Neg'])
        self.performance_measures['Precision'] = self.performance_measures['TP'] / (self.performance_measures['TP'] + self.performance_measures['FP'])
        self.performance_measures['Recall'] = self.performance_measures['TP'] / self.performance_measures['Pos']

    def img_indices(self):
        '''Get the indices of true and false positives to be able to locate the corresponding images in a list of image names'''
        self.performance_df['tp_ind'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == True))
        self.performance_df['fp_ind'] = ((self.performance_df['preds'] == True) & (self.performance_df['labls'] == False))
        self.image_indices['TP_indices'] = np.where(self.performance_df['tp_ind']==True)[0].tolist()
        self.image_indices['FP_indices'] = np.where(self.performance_df['fp_ind']==True)[0].tolist()

# convert the lists to ndarrays
features = np.asarray(features_list)
Y = np.asarray(y_list)
imgs = np.asarray(imnames_list)
print(features.shape)

# flatten the images ndarray to one row per image
features_flat = features.reshape((features.shape[0], -1))
print(features_flat.shape)
print(Y.shape)
print(imgs.shape)

# create train and test sets
from sklearn.cross_validation import train_test_split

data_train, data_test, y_train, y_test, imgs_train, imgs_test = train_test_split(features_flat, 
    Y, imgs, test_size = 0.5, random_state = 71)

# MODEL: Perceptron
from sklearn import linear_model
prc = linear_model.SGDClassifier(loss='perceptron')
prc.fit(data_train, y_train)

prc_performance = BinaryClassificationPerformance(prc.predict(data_train), y_train, 'prc')
prc_performance.compute_measures()
print(prc_performance.performance_measures)

prc_performance_test = BinaryClassificationPerformance(prc.predict(data_test), y_test, 'prc')
prc_performance_test.compute_measures()
print(prc_performance_test.performance_measures)

prc_performance_test.img_indices()
img_indices_to_view = prc_performance_test.image_indices

# look at the true positives in the test set
for i in range(len(img_indices_to_view['TP_indices'])):
    toRead = subdirectory + imgs_test[img_indices_to_view['TP_indices'][i]]
    img_raw = io.imread(toRead)
    plt.figure()
    io.imshow(img_raw)

# look at the false positives in the test set
for i in range(len(img_indices_to_view['FP_indices'])):
    toRead = subdirectory + imgs_test[img_indices_to_view['FP_indices'][i]]
    img_raw = io.imread(toRead)
    plt.figure()
    io.imshow(img_raw)

# MODEL: Multi-layer Perceptron aka neural network
from sklearn import neural_network
nn = neural_network.MLPClassifier()
nn.fit(data_train, y_train)

nn_performance = BinaryClassificationPerformance(nn.predict(data_train), y_train, 'nn')
nn_performance.compute_measures()
print(nn_performance.performance_measures)

nn_performance_test = BinaryClassificationPerformance(nn.predict(data_test), y_test, 'nn_test')
nn_performance_test.compute_measures()
print(nn_performance_test.performance_measures)



