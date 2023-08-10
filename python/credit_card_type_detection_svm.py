# Loading required libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
df = pd.read_csv('input/credit-card.csv')
# Since other card types have too little sample, we only use 3 types of cards as our study target
df = df[df.card_type.isin(['Visa', 'MasterCard', 'UnionPay'])] 

# Load the image
from scipy import misc

card_images = list(map(lambda img: misc.imread('input/train_trim_card_image/'+img)[:,:,:3], df.image_name))
# Reshape the array into single dimension
card_image_array = list(map(lambda img: img.ravel(), card_images))
# Transform list of array into matrix
card_image_matrix = np.vstack(card_image_array)

# We have (no. of image) rows and (width x height x RGB Channels) column
card_image_matrix.shape

# As usual practice, we split the data into 80% train and 20% test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(card_image_matrix, df.card_type, test_size=0.2, random_state=812)

# we will first PCA to reduce the matrix dimension, 
# noticing that some feature such as background color can be reduced to low dimensionality since every image has the same layout.
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(X_train)
print(np.cumsum(pca.explained_variance_ratio_)) # We see that 50 components already explained 95% of variablility

# Next we transform the original train and test data into low dimension (50) data
X_train_reduced = pca.transform(X_train)
X_test_reduced = pca.transform(X_test)

# Now, Here we go the most exciting part: Multiclass SVM with linear kernel
from sklearn.svm import SVC
clf = SVC(kernel ='linear', max_iter=10000)
clf.fit(X_train_reduced, y_train) 
clf.score(X_train_reduced, y_train) # It predict perfectly on training set!

clf.score(X_test_reduced, y_test) # With good cv score.

prediction = pd.Series(clf.predict(X_test_reduced), index = y_test.index, name='prediction')
df_prediction = pd.concat((df, prediction), axis=1, join='inner')
df_prediction_correct = df_prediction.loc[lambda df:df.card_type==df.prediction, :]
df_prediction_wrong = df_prediction.loc[lambda df:df.card_type!=df.prediction, :]
df_prediction_wrong

# Plot some of the correct prediction
import random
random.seed(9)
plt.figure(figsize=(16,3))
for i in range(4):
    idx = random.randint(0, len(df_prediction_correct.index)-1)
    plt.subplot(1,4,i+1).set_title(df_prediction_correct['prediction'].iloc[idx], fontsize=24)
    plt.imshow(misc.imread('input/train_raw_card_image/'+df_prediction_correct.image_name.values[idx]))
    plt.axis('off')

# Plot the wrong prediction
num_wrong = len(df_prediction_wrong.index)
plt.figure(figsize=(8,3))
for i in range(num_wrong):
    print(df_prediction_wrong.name.values[i])
    plt.subplot(1,num_wrong,i+1).set_title(df_prediction_wrong['prediction'].iloc[i], fontsize=24)
    plt.imshow(misc.imread('input/train_raw_card_image/'+df_prediction_wrong.image_name.values[i]))
    plt.axis('off')

# Function to predict any other single image
def predict_credit_card_type(img_file):
    img=misc.imread(img_file)[:,:,:3]
    img = misc.imresize(img, (56, 92, 3)).ravel()
    img = np.reshape(img, (1, -1))
    img = pca.transform(img)
    return clf.predict(img)

test_images = ['input/test_card_image/'+x for x in ['master-card.jpg', 'visa.jpg','mix.jpg', 'master-card-2.jpg', 'test.jpg']]
num_test_images = len(test_images)
plt.figure(figsize=(16,6))
for i in range(num_test_images):
    plt.subplot(2,num_test_images//2+1,i+1).set_title(predict_credit_card_type(test_images[i])[0], fontsize=24)
    plt.imshow(misc.imread(test_images[i]))
    plt.axis('off')

