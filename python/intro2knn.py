# Set up Notebook

get_ipython().magic('matplotlib inline')

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We do this to ignore several specific Pandas warnings
import warnings
warnings.filterwarnings("ignore")

# We import helper code for getting and plotting the Iris data
from helper_code import iris as hi

# Now we grab the raw data, and also create a sampled grid of data.

# This code restricts the features to the top two PCA components
# This simplifies the viewing of the predited labels.

data = hi.get_data()
mdata = hi.get_mdata(data)

# Get features (x) and labels (y)
x = data[:, 0:2]
y = data[:, 2]

# Show the data
cols = ['PCA1', 'PCA2', 'Species']

# We make a plot of the features.
hi.scplot_data('PCA1', 'PCA2', pd.DataFrame(data, columns = cols), 'Species',
               'First PCA', 'Second PCA', (-4.2, 4.6), (-1.8, 1.6), 6)

import sklearn.cross_validation as cv
(x_train, x_test, y_train, y_test) = cv.train_test_split(x, y, test_size=.25)

from sklearn import neighbors as nb

# The number of neighbors affects performance
nbrs = 3

# First we construct our Classification Model
knc = nb.KNeighborsClassifier(n_neighbors=nbrs)
knc.fit(x_train, y_train);

z = knc.predict(mdata)

hi.splot_data(data, mdata, z, 'First Component', 'Second Component', 50)

print("KNN ({0} neighbors) prediction accuracy = {1:5.1f}%".format(nbrs, 100.0 * knc.score(x_test, y_test)))

from sklearn.metrics import classification_report

y_pred = knc.predict(x_test)
print(classification_report(y_test, y_pred,                             target_names = ['Setosa', 'Versicolor', 'Virginica']))

from helper_code import mlplots as mlp

mlp.confusion(y_test, y_pred, ['Setosa', 'Versicolor', 'Virginica'], 3, "KNN-({0}) Model".format(nbrs))

from helper_code import digits as hd

x, y, images = hd.get_data()
hd.im_plot(x, y, images)

print('Total number of samples = {0}'.format(y.shape[0]))

(x_train, x_test, y_train, y_test) = cv.train_test_split(x, y, test_size=.25)

knc = nb.KNeighborsClassifier()
knc.fit(x_train, y_train);
print('Prediction Accuracy = {0:3.1f}%'.format(100*knc.score(x_test, y_test)))

y_pred = knc.predict(x_test)
print(classification_report(y_test, y_pred))

nms = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

mlp.confusion(y_test, y_pred, nms, 10, "KNN-({0}) Model".format(nbrs))

# Test on our sevens

ones = hd.make_ones()

hd.plot_numbers(ones)

# You can change the values to make other numbers.

ones[0].reshape(8,8)

print('Actual : Predicted')

for one in ones:
    print('  1    :     {0}'.format(knc.predict(one.ravel().reshape(1, -1))[0])) 

# Now test on our sevens

sevens = hd.make_sevens()
hd.plot_numbers(sevens)

print('Actual : Predicted')
for seven in sevens:
    print('  7    :     {0}'.format(knc.predict(seven.ravel().reshape(1, -1))[0])) 

