# Display plots in the notebook
get_ipython().magic('matplotlib inline')

import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.datasets import load_digits

# Load all the samples for all digits 0-9
digits = load_digits()

# Assign the matrices to a variable `data`
data = digits.data

# Assign the labels to a variable `target`
target = digits.target

from sklearn.decomposition import PCA

# Create the PCA model
pca = PCA()

# Fit the model to our data and then project the data onto the components
Xproj = pca.fit_transform(data)

# Create a figure
figure, ax = plt.subplots(figsize=(10,10))

# Scatter the projections of the first and second principal components and color them by their labels
s = ax.scatter(Xproj[:, 0], Xproj[:, 1], c=target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_rainbow', 10))

# Create a colorbar for reference
figure.colorbar(s, ax=ax)

from sklearn.preprocessing import scale

figure, (axs) = plt.subplots(figsize=(16,1), ncols=16)
for i,ax in enumerate(axs):
    # Restore the first i components of the reduced data
    # This involves multiplying the projection by the principal components,
    # adding up the values and then adding the mean for each pixel.
    digit = np.sum(pca.components_[:i] * Xproj[0][:i][:,np.newaxis], axis=0) + pca.mean_
    
    # Show an "image" - a nxn array of values
    ax.imshow(digit.astype(int).reshape((8,8)), interpolation="nearest", cmap="Greys")
    ax.set_axis_off()
    ax.set_title(i)

from sklearn.decomposition import KernelPCA

# Create the Kernel PCA model with a 4th degree polynomial kernel, gamma of .009 and coef0 of 120
pca = KernelPCA(kernel='poly', degree=4, gamma=.009, coef0=120)

# Fit the model to our data and then project the data onto the components
Xproj = pca.fit_transform(data)

# Create a figure
figure, ax = plt.subplots(figsize=(10,10))

# Scatter the projections of the first and second principal components and color them by their labels
s = ax.scatter(Xproj[:, 0], Xproj[:, 1], c=target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('gist_rainbow', 10))

# Create a colorbar for reference
figure.colorbar(s, ax=ax)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

# Use the first 8 dimensions of the reduced data to create a train/test split
data_train, data_test, target_train, target_test = train_test_split(
    Xproj[:,:8], target, test_size=.25, random_state=0
)

# Create the model as we did before
model = DecisionTreeClassifier(max_depth=8)

# Fit it to our reduced data
model.fit(data_train, target_train)

# Use the model to predict labels for our training set
pred_train = model.predict(data_train)

# And for the test set
pred_test = model.predict(data_test)

# Print the accuracy for the training set
print("Training Accuracy:", accuracy_score(target_train, pred_train))

# Print the accuracy for the test set
print("Testing Accuracy:", accuracy_score(target_test, pred_test))

# Use the original to create a train/test split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=.25, random_state=0
)

# Create the model as we did before
model = DecisionTreeClassifier(max_depth=8)

# Fit it to our reduced data
model.fit(data_train, target_train)

# Use the model to predict labels for our training set
pred_train = model.predict(data_train)

# And for the test set
pred_test = model.predict(data_test)

# Print the accuracy for the training set
print("Training Accuracy:", accuracy_score(target_train, pred_train))

# Print the accuracy for the test set
print("Testing Accuracy:", accuracy_score(target_test, pred_test))



