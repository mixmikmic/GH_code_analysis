import numpy as np
import matplotlib as mp
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Load the sample data set from the datasets module
dataset = datasets.load_iris()

# Display the data in the test dataset
dataset

# Species of Iris in the dataset
dataset['target_names']

# Names of the type of information recorded about an Iris - called features
dataset['feature_names']

# First 10 sets of Iris data
dataset['data'][:10]

# The classification of each of the first 10 sets of Iris data - the target
dataset['target'][:10]

# Now we create our model
model = LogisticRegression()
# We train it by passing in the test data and the actual results
model.fit(dataset.data, dataset.target)

# We use the model to create predictions
expected = dataset.target
predicted = model.predict(dataset.data)
# Using the metrics module we see the results of the model
metrics.accuracy_score(expected, predicted, normalize=True, sample_weight=None)

y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird"]

metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)

print(metrics.classification_report(y_true, y_pred,
    target_names=["ant", "bird", "cat"]))

metrics.confusion_matrix(y_true, y_pred)

print(metrics.classification_report(expected, predicted,target_names=dataset['target_names']))

print (metrics.confusion_matrix(expected, predicted))

