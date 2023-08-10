#Only for generating data
from sklearn.datasets import make_blobs

help(make_blobs)

x,y = make_blobs(n_samples=300, centers=4, n_features=2,random_state=0, cluster_std=1.0)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.scatter(x[:,0], x[:,1], c=y, s=50,cmap='rainbow')

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(x,y)

x_test,y_test = make_blobs(n_samples=30000, centers=4, n_features=2,random_state=0, cluster_std=2.0)

y_pred = tree.predict(x_test)

y_pred


plt.scatter(x_test[:,0], x_test[:,1], c=y_pred, s=50,cmap='rainbow', alpha=0.05)

tree.predict([[10.0,0]])

y


from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()

# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

digits.data[0].reshape((8,8))

digits.data.shape

#Split data into train & test sets
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state=0)

Xtrain.shape

Xtest.shape

#help(train_test_split)

from sklearn.ensemble import RandomForestClassifier
#n_estimators - Num of decison trees
model = RandomForestClassifier(n_estimators=1000)

model.fit(Xtrain, ytrain)

ypred = model.predict(Xtest)

from sklearn import metrics

print(metrics.classification_report(ypred, ytest))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set()

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

model.predict([Xtest[0]])

Xtest[0].reshape((8,8))

fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.brg, interpolation='nearest')
    
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))



