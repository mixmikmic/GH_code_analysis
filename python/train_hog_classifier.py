import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from hog_util_functions import *
from sklearn.utils import shuffle

from train_hog_classifier import load_data_sets
cars, notcars = load_data_sets()

print('Number of samples in cars set: ', len(cars))
print('Number of samples in notcars set: ', len(notcars))

params = {}

params['color_space'] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
params['orient'] = 9  # HOG orientations
params['pix_per_cell'] = 8 # HOG pixels per cell
params['cell_per_block'] = 2 # HOG cells per block
params['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
params['spatial_size'] = (32, 32) # Spatial binning dimensions
params['hist_bins'] = 32    # Number of histogram bins
params['spatial_feat'] = True # Spatial features on or off
params['hist_feat'] = True # Histogram features on or off
params['hog_feat'] = True # HOG features on or off

from train_hog_classifier import extract_features

t1=time.time()

cars_feats = extract_features(cars, params)
notcars_feats = extract_features(notcars, params)

t2 = time.time()
print(round(t2-t1, 2), 'second to extract features (HOG,spatial and color features).')

assert(len(cars_feats) == len(cars))
assert(len(notcars_feats) == len(notcars))

# Create an array stack of feature vectors
X = np.vstack((cars_feats, notcars_feats)).astype(np.float64)

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)

# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Number of samples in train set: ', len(X_train))
print('Number of samples in test set: ', len(X_test))

from train_hog_classifier import save_classifier_data
#save_classifier_data('HOGClassifierData.p', X_train, y_train, X_test, y_test)

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 100
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

from train_hog_classifier import save_classifier
save_classifier('HOGClassifier.p', svc, X_scaler, params)

car_1, car_2 = 30, 5000    # indexes if two car samples
ncar_1, ncar_2 = 500, 5100 # indexes if two non-car samples

# Must be equal to params['color_space'] !!!!!!!!!!!!!!!!!!!!!!
colorspace = cv2.COLOR_RGB2YCrCb

font_size = 15
f, axarr = plt.subplots(4, 7,figsize=(20,10))
#f.subplots_adjust(hspace=0.2, wspace=0.05)

for i, j in enumerate([car_1, car_2]):
    image = plt.imread(cars[j])
    feature_image = cv2.cvtColor(image, colorspace)

    axarr[i,0].imshow(image)
    axarr[i,0].set_xticks([])
    axarr[i,0].set_yticks([])
    title = "car {0}".format(j)
    axarr[i,0].set_title(title, fontsize=font_size)

    for channel in range(3):        
        axarr[i, channel+1].imshow(feature_image[:,:,channel],cmap='gray')
        title = "ch {0}".format(channel)
        axarr[i,channel+1].set_title(title, fontsize=font_size)
        axarr[i,channel+1].set_xticks([])
        axarr[i,channel+1].set_yticks([])    
    
    for channel in range(3):
        features,hog_image = get_hog_features(feature_image[:,:,channel], params['orient'], params['pix_per_cell'], 
                                              params['cell_per_block'], vis=True, feature_vec=True)
        axarr[i,channel+4].imshow(hog_image,cmap='gray')
        title = "HOG ch {0}".format(channel)
        axarr[i,channel+4].set_title(title, fontsize=font_size)
        axarr[i,channel+4].set_xticks([])
        axarr[i,channel+4].set_yticks([])

for k, j in enumerate([ncar_1, ncar_2]):
    i=k+2
    image = plt.imread(notcars[j])
    feature_image = cv2.cvtColor(image, colorspace)

    axarr[i,0].imshow(image)
    axarr[i,0].set_xticks([])
    axarr[i,0].set_yticks([])
    title = "not car {0}".format(j)
    axarr[i,0].set_title(title, fontsize=font_size)

    for channel in range(3):        
        axarr[i,channel+1].imshow(feature_image[:,:,channel],cmap='gray')
        title = "ch {0}".format(channel)
        axarr[i,channel+1].set_title(title, fontsize=font_size)
        axarr[i,channel+1].set_xticks([])
        axarr[i,channel+1].set_yticks([])        
    
    for channel in range(3):
        features,hog_image = get_hog_features(feature_image[:,:,channel], params['orient'], params['pix_per_cell'], 
                                              params['cell_per_block'], vis=True, feature_vec=True)
        axarr[i,channel+4].imshow(hog_image,cmap='gray')
        title = "HOG ch {0}".format(channel)
        axarr[i,channel+4].set_title(title, fontsize=font_size)
        axarr[i,channel+4].set_xticks([])
        axarr[i,channel+4].set_yticks([])
        
plt.show()

def plot_example_raw_and_scaled_features(title, img_file, feature, norm_feature):
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(img_file))
    plt.title('Original Image: ' + title)
    plt.subplot(132)
    plt.plot(feature)
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(norm_feature)
    plt.title('Normalized Features')
    fig.tight_layout()
    plt.show()
    pass
   
if len(X) > 0:
    plot_example_raw_and_scaled_features("car {0}".format(car_1), cars[car_1], X[car_1], scaled_X[car_1])
    plot_example_raw_and_scaled_features("car {0}".format(car_2), cars[car_2], X[car_2], scaled_X[car_2])
    plot_example_raw_and_scaled_features("not car {0}".format(ncar_1), notcars[ncar_1], X[len(cars)+ncar_1], scaled_X[len(cars)+ncar_1])
    plot_example_raw_and_scaled_features("not car {0}".format(ncar_2), notcars[ncar_2], X[len(cars)+ncar_2], scaled_X[len(cars)+ncar_2])
else:
    print('X is empty. Calculate X above!')

from find_cars import find_cars
get_ipython().magic('matplotlib inline')

dist_pickle = pickle.load( open("HOGClassifier.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]

params = {}
params['color_space']    = dist_pickle['color_space']
params['orient']         = dist_pickle['orient']
params['pix_per_cell']   = dist_pickle['pix_per_cell']
params['cell_per_block'] = dist_pickle['cell_per_block']
params['hog_channel']    = dist_pickle['hog_channel']
params['spatial_size']   = dist_pickle['spatial_size']
params['hist_bins']      = dist_pickle['hist_bins']
params['spatial_feat']   = dist_pickle['spatial_feat']
params['hist_feat']      = dist_pickle['hist_feat']
params['hog_feat']       = dist_pickle['hog_feat']

print(params)

#img = mpimg.imread('test_images/test1.jpg')
#img = mpimg.imread('test_images/test2.jpg')
#img = mpimg.imread('test_images/test3.jpg')
img = mpimg.imread('test_images/test4.jpg')
#img = mpimg.imread('test_images/test5.jpg')
#img = mpimg.imread('test_images/test6.jpg')

#img = mpimg.imread('temp_data/frames/project_video/718.jpg')

if img.shape[2] > 3:
    img = img[:,:,0:3]

ystart = 400
ystop = 656
scale = 1.5
    
draw_img = np.copy(img)
img = img.astype(np.float32)/255

t=time.time()

bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, params)

t2 = time.time()
print(round(t2-t, 5), 'Seconds to detect using scale:', scale)

draw_img = draw_boxes(draw_img, bboxes, thick=4)

fig = plt.figure(figsize=(20,10))
plt.imshow(draw_img)
plt.title('Hot windows')
plt.show()



