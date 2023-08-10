# This tells matplotlib not to try opening a new window for each plot.
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

print "Keys", boston.keys()

# It might also be useful to see what our other feature names are 
print "Feature names", boston['feature_names']

# In this data 'data' are the features and 'target' is the median home value
features = boston['data']
medianhome = boston['target']

# Let's see the distribution of the median home value per neighborhood
#plt.hist(medianhome)

# Divide 'medianhome' by high and low cost housing
median = np.median(medianhome)

# Create a recoded 'medianhome' variable
medianclass = np.where(medianhome > median, 1, 0)

print "Recoded medianhome variable", medianclass

# What are the dimensions of the data? 
print 'Feature matrix',  np.shape(features)

print 'Class label' , len(medianclass)

# Prep the data for training
X, Y = features, medianclass

# Shuffle the data, but make sure that the features and accompanying labels stay in sync.
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

# Split into train and test.
train_data, train_labels = X[:400], Y[:400]
test_data, test_labels = X[400:], Y[400:]

def EuclideanDistance(v1, v2):
    sum = 0.0
    for index in range(len(v1)):
        sum += (v1[index] - v2[index]) ** 2
    return sum ** 0.5

dists = []
for i in range(len(train_data) - 1):
    for j in range(i + 1, len(train_data)):
        dist = EuclideanDistance(train_data[i], train_data[j])
        dists.append(dist)
        
fig = plt.hist(dists, 100)

class NearestNeighbors:
    # Initialize an instance of the class.
    def __init__(self, metric=EuclideanDistance):
        self.metric = metric
    
    # No training for Nearest Neighbors. Just store the data.
    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
    
    # Make predictions for each test example and return results.
    def predict(self, test_data):
        results = []
        for item in test_data:
            results.append(self._predict_item(item))
        return results
    
    # Private function for making a single prediction.
    def _predict_item(self, item):
        best_dist, best_label = 1.0e10, None
        for i in range(len(self.train_data)):
            dist = self.metric(self.train_data[i], item)
            if dist < best_dist:
                best_label = self.train_labels[i]
                best_dist = dist
        return best_label

clf = NearestNeighbors()
clf.fit(train_data, train_labels)
preds = clf.predict(test_data)

correct, total = 0, 0
for pred, label in zip(preds, test_labels):
    if pred == label: correct += 1
    total += 1
print 'total: %3d  correct: %3d  accuracy: %3.2f' %(total, correct, 1.0*correct/total)

# Let's make a confusion matrix to see how many true and false positives we have
from sklearn.metrics import confusion_matrix

true = test_labels
predicted = preds

confusion_matrix(true, predicted)

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cnf_matrix = confusion_matrix(true, predicted)
np.set_printoptions(precision=2)

class_names = ["High Income", "Low Income"]

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')






