import scipy.io
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# load our dataset
train_data = scipy.io.loadmat('extra_32x32.mat')

# extract the images and labels from the dictionary object
X = train_data['X']
y = train_data['y']

# view an image (e.g. 10) and print its corresponding label
img_index = 10
plt.imshow(X[:,:,:,img_index])
print(y[img_index])

from sklearn.utils import shuffle

X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
y = y.reshape(y.shape[0],)
X, y = shuffle(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

print(clf)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

preds = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test,preds))



