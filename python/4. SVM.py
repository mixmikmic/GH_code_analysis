from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC

# load the iris datasets
from sklearn import svm, datasets
dataset = datasets.load_iris()

# fit a SVM model to the data

model = svm.SVC(kernel='linear')
model.fit(dataset.data, dataset.target)

# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)

from sklearn import metrics

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

