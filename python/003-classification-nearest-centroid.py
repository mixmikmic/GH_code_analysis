import numpy as np

# global variable
global img_number
img_number = 700

output_filename = 'digit_fv.train'
digit_data = np.loadtxt(output_filename, dtype='float64')

print("Size of the feature vector", digit_data.shape)
print digit_data[0:10,0:]

import math
from skimage import io

# show image
def vec2img(vec):
    
    img_row = int(math.sqrt(vec.shape[0]))
    img_col = img_row
    img = vec.reshape((img_row, img_col))
    
    io.imshow(img)
    io.show()
    
# check predict
def chkpredict(actual_class, predict_class):
    if(actual_class==predict_class):
        print("(Correct prediction)")
    else:
        print("(Incorrect prediction)")

# show image
vec2img(digit_data[img_number,1:])

from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np

X = digit_data[:,1:]
y = digit_data[:,0]

print("Size of feature vector", X.shape)
print("Size of label", y.shape)

# create a model
clf = NearestCentroid()
clf.fit(X, y)

print("Actual", y[img_number])
predict_class = clf.predict([X[img_number,:]])
print("Predict", predict_class)
chkpredict(y[img_number], predict_class)

# KNN
from sklearn import neighbors

# Classifier implementing the k-nearest neighbors vote.
knn = neighbors.KNeighborsClassifier()    
    # default: n_neighbors=5, weights='uniform'

# we create an instance of Neighbours Classifier and fit the data.
knn.fit(X, y)

print("Actual", y[img_number])
predict_class = knn.predict([X[img_number,:]])
print("Predict", predict_class)
chkpredict(y[img_number], predict_class)

# KNN
from sklearn import neighbors

# Classifier implementing the k-nearest neighbors vote.
knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')

# we create an instance of Neighbours Classifier and fit the data.
knn.fit(X, y)

print("Actual", y[img_number])
predict_class = knn.predict([X[img_number,:]])
print("Predict", predict_class)
chkpredict(y[img_number], predict_class)



