get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import os
from time import time
import numpy as np
from data import get_hsi_data

# get data
img = get_hsi_data.get_data()

# reshape image as array
from utils.image import img_as_array
img_Vec = img_as_array(img['original'])
img_gt = img_as_array(img['groundtruth'], gt=True)

# normalize the data
from utils.image import standardize
img_Vec = standardize(img_Vec)

# pair x and y with gt
from utils.image import img_gt_idx
X, y = img_gt_idx(img_Vec, img_gt, printinfo=True)

# get training and testing samples
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=.10)

from manifold_learning.lpp import LocalityPreservingProjections

t0 = time()

lpp_model = LocalityPreservingProjections(n_neighbors=20,
                                          neighbors_algorithm='brute',
                                          n_components=50,
                                          sparse=True,
                                          eig_solver='dense')
lpp_model.fit(X_train)
X_proj_train = lpp_model.transform(X_train)
X_proj_test = lpp_model.transform(X_test)
eigVals = lpp_model.eigVals
t1 = time()

print('My LPP class: {t:.2g} secs'.format(t=t1-t0))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

fig, ax = plt.subplots()

ax.plot(eigVals)
plt.show()

# classification using LDA
from sklearn.lda import LDA

lda_model = LDA()
lda_model.fit(X_proj_train,y_train)
y_pred = lda_model.predict(X_proj_test)

# classification report
from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


print(cr)

plt.figure(figsize=(15,15))
plt.matshow(cm,aspect='auto')
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# predict the entire HSI image
t0 = time()
img_Vec_proj = lpp_model.transform(img_Vec)
print(np.shape(img_Vec_proj))
lda_image = lda_model.predict(img_Vec_proj)
t1 = time()

# reshape the image
lda_image = lda_image.reshape(img['original'][:,:,0].shape)


# Look at Ground Truth for Indian Pines
fig, ax = plt.subplots()
h = ax.imshow(lda_image, cmap=plt.cm.jet)
# colorbar_ax = fig.add_axes([0, 16, 1, 2])
fig.colorbar(h, ticks=[np.linspace(1,16,num=16,endpoint=True)])
plt.title('Indian Pines - Predicted Image')
plt.show()

# jakes algorithm
from lpproj import LocalityPreservingProjection
lpp = LocalityPreservingProjection(n_components=50)

t0 = time()
X_proj_train = lpp.fit_transform(X_train)
X_proj_test = lpp.transform(X_test)
t1=time()
print('My LPP class: {t:.2g} secs'.format(t=t1-t0))

# classification using LDA
from sklearn.lda import LDA

lda_model = LDA()
lda_model.fit(X_proj_train,y_train)
y_pred = lda_model.predict(X_proj_test)

# classification report
from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


print(cr)

plt.figure(figsize=(15,15))
plt.matshow(cm,aspect='auto')
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()



