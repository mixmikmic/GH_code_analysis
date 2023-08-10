import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.model_selection import train_test_split
import pandas as pd

# Only 2 classes

data = pd.read_csv('wine_original.csv')
data = data[data['class'] != 3]
labels = data['class']
del data['class']

data.head()

# Split into testing and training data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=5)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

parameters = { 'penalty': ['l1','l2'], 
              'C':[0.1, 0.5, 1, 2, 3, 4, 5, 10]}
logreg = LogisticRegression()
clf = GridSearchCV(logreg, parameters, verbose=True, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
train_acc = accuracy_score(clf.predict(X_train), y_train)
print ('Selected Parameters: ', clf.best_params_)
print ('Training Accuracy = ' + str(train_acc))
print ('Test Accuracy = ' + str(accuracy))

clf.coef_

from matplotlib.colors import ListedColormap

h = .008  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


clf = LogisticRegression(C=1000, penalty='l1')
clf.fit(X_train[['Alcohol', 'Malic acid']], y_train)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train['Alcohol'].min() - 1, X_train['Alcohol'].max() + 1
y_min, y_max = X_train['Malic acid'].min() - 1, X_train['Malic acid'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training & test points
plt.scatter(X_train['Alcohol'], X_train['Malic acid'], c=y_train, cmap=cmap_bold)
plt.scatter(X_test['Alcohol'], X_test['Malic acid'], c=y_test,marker='x', cmap=cmap_bold)
plt.xlabel('Alcohol', fontsize=15)
plt.ylabel('Malic Acid', fontsize=15)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

clf.coef_

# C = 0.0001

clf = LogisticRegression(C=0.0001, penalty='l1')
clf.fit(X_train[['Alcohol', 'Malic acid']], y_train)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train['Alcohol'].min() - 1, X_train['Alcohol'].max() + 1
y_min, y_max = X_train['Malic acid'].min() - 1, X_train['Malic acid'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training & test points
plt.scatter(X_train['Alcohol'], X_train['Malic acid'], c=y_train, cmap=cmap_bold)
plt.scatter(X_test['Alcohol'], X_test['Malic acid'], c=y_test,marker='x', cmap=cmap_bold)
plt.xlabel('Alcohol', fontsize=15)
plt.ylabel('Malic Acid', fontsize=15)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

clf.coef_

# No penalty
clf = LogisticRegression(penalty='l1', C=100000000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

#original weights
orig_weights = clf.coef_
print ('original weights: ')
print (orig_weights)

# Regularization l1
clf = LogisticRegression(penalty='l1', C=0.5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

print ('lasso weights: ')
print (clf.coef_)

# Regularization l2
clf = LogisticRegression(penalty='l2', C=0.5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)*1.0/len(y_test)))

print ('ridge weights: ')
print (clf.coef_)

# Why l1 gives a sparse weight vector

# Regularization l2, varying C: Low C
clf = LogisticRegression(penalty='l2', C=0.000000001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))

clf.coef_

clf = LogisticRegression(penalty='l2', C=0.0001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))

clf.coef_

# Good C
clf = LogisticRegression(penalty='l2', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))

clf.coef_

# Effect of number of iterations on accuracy
result=[]
for iter_cnt in range(1, 40):
    clf = LogisticRegression(penalty='l2', solver= 'newton-cg', max_iter=iter_cnt)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result.append(1-np.sum(y_pred == y_test)/len(y_test))

# Plot
plt.figure(figsize=(15,10))
plt.title('Error vs Iterations', fontsize=15)
plt.xlabel('Number of Iterations', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.plot(range(1, 40), result)
plt.show()

# Gradient Descent
from sklearn.linear_model import SGDClassifier

result = []

# small learning rate
for iter_cnt in range(1, 40):
    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.001, learning_rate='constant', eta0=0.00000001, verbose=0, random_state=7, n_iter=iter_cnt)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result.append(1-np.sum(y_pred == y_test)*1.0/len(y_test))
#     print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))


# Plot
plt.figure(figsize=(15,10))
plt.title('Error vs Iterations', fontsize=15)
plt.xlabel('Number of Iterations', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.plot(range(1, 40), result)
plt.show()

clf.coef_

y_pred

# good learning rate

result = []

for iter_cnt in range(1, 40):
    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.001, learning_rate='constant', eta0=0.01, verbose=0, random_state=7, n_iter=iter_cnt)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result.append(1-np.sum(y_pred == y_test)/len(y_test))
#     print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))


# Plot
plt.figure(figsize=(15,10))
plt.title('Error vs Iterations', fontsize=15)
plt.xlabel('Number of Iterations', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.plot(range(1, 40), result)
plt.show()

clf.coef_

# high learning rate

result = []

for iter_cnt in range(1, 40):
    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.001, learning_rate='constant', eta0=10, verbose=0, random_state=7, n_iter=iter_cnt)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result.append(1-np.sum(y_pred == y_test)/len(y_test))
#     print ('Test accuracy = ' + str(np.sum(y_pred == y_test)/len(y_test)))


# Plot
plt.figure(figsize=(15,10))
plt.title('Error vs Iterations', fontsize=15)
plt.xlabel('Number of Iterations', fontsize=15)
plt.ylabel('Error', fontsize=15)
plt.plot(range(1, 40), result)
plt.show()

clf.coef_



