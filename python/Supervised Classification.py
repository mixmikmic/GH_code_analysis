# You don't want to change anything here now

import numpy as np   # For some numerical stuff
import matplotlib.pyplot as plt # For making beautiful plots
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier  # A simple machine learning model known as KNN
from sklearn.cross_validation import train_test_split # A utility to split data
from sklearn.metrics import precision_score
get_ipython().magic('pylab inline')

# You may see some messages in the next line, don't worry about them

dataset = load_iris() # Load the complete iris data structure to this variable

# Now lets get the features
features = dataset['data']

# Lets also get the name of the features
feature_names = dataset['feature_names']

# The class labels
labels = dataset['target']


# Lets have a look at the names of the features and dimensions (shape) of the feature array and also see how many classes are present.
# Verify if the number of feature names are equal to the number of columns

print 'Feature names are :', feature_names

print '\nThe feature array has %d rows and %d columns'%(features.shape[0],features.shape[1])

print '\nThere are %d classes of objects in the dataset'%(len(np.unique(labels)))

index_1 = 0 # Modify this to change the x-axis . Now it will take the first column. [In python index 1 starts at '0']
index_2 = 1 # Modify this to change the y-axis

plt.scatter(features[:,index_1],features[:,index_2],c=labels) # Make the scatter plot
plt.xlabel(feature_names[index_1])
plt.ylabel(feature_names[index_2])

# train_data --> feature samples for training
# test_data  --> feature samples to evaluate / test
# train_labels --> class labels for the training data
# test_labels --> class labels for the test data

train_data,test_data,train_labels,test_labels = train_test_split(features,labels,test_size=0.3,random_state=0)

# Lets have a look at the size of the train and test data

print 'Train data has %d samples'%(train_data.shape[0])
print 'Test data has %d samples'%(test_data.shape[0])

mymodel = KNeighborsClassifier(n_neighbors=5,)  # Create the classifier object to a variable 'mymodel'

mymodel = mymodel.fit(train_data,train_labels) # Train the algorithm and save the model mymodel 

# Test the performance of the algorithm on the test data which was generated through the splitting before.

predictions = mymodel.predict(test_data)

# Now we have the class labels predicted by the algorithm for each test samples in the variable 'predictions'


# Time to check the precision score

score = precision_score(predictions,test_labels,average='micro')

print 'The precision score is %f'%(score*100)

