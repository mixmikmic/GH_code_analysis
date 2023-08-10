get_ipython().magic('matplotlib inline')

from matplotlib.pyplot import imshow, plot, subplot, figure
from scipy.ndimage import uniform_filter
from scipy.misc import imread
import dask.array as da
import numpy as np

get_ipython().system('if [ ! -e mnist.pkl.gz ]; then wget -q https://github.com/michhar/python-jupyter-notebooks/raw/master/bigdata/images/mnist.pkl.gz; fi')

get_ipython().system('ls -lh')

import pickle, gzip
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='bytes')

digit_idx = 0 # change to check other digits

img = train_set[0][digit_idx].reshape(28, 28)
img = (img*255).astype(np.uint8) # image read as float32, image is 8 bit grayscale
imshow(img)

num_pix = str(img.shape[0] * img.shape[1])
print('%s pixels, shape %s, dtype %s (%d total digit images)' % (num_pix, img.shape, img.dtype, len(train_set[0])))
print('Label: %d' % train_set[1][digit_idx])

X_train = da.from_array(train_set[0], chunks=(1000, 1000))
y_train = da.from_array(train_set[1], chunks=(1000, 1000))

X_test = da.from_array(test_set[0], chunks=(1000))
y_test = da.from_array(test_set[1], chunks=(1000))

from sklearn.linear_model import SGDClassifier

# Instatiate
sgd = SGDClassifier()

# Fit
get_ipython().magic('time sgd = da.learn.fit(sgd, X_train, y_train, classes=range(0, 10))')

# Dask.learn predict method
y_pred = da.learn.predict(sgd, X_test)

# Convert back to an array by slicing and check first 10 predictions
y_pred[:10].compute()

# Check the actual labels - do they agree?
y_test[:10].compute()

