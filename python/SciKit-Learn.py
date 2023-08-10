from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC 
# svm - Support Vector Machine
# SVC - Support Vector Classifier

ds = datasets.load_iris() #Loading the in-built dataset

ds.feature_names

ds.target_names

model = SVC() #Using pre-built model in SciKit-Learn
model.fit(ds.data, ds.target) #Fit a SVM Model to the data
print(model)

expected = ds.target

#Make Prediction
predicted = model.predict(ds.data)

#Summarize the fit of the model
print(metrics.classification_report(expected, predicted))

#Print Accuracy
print(metrics.accuracy_score(expected, predicted))

#Print number of correctly classified samples
print(metrics.accuracy_score(expected, predicted, normalize=False))

# Confusion Matrix
print(metrics.confusion_matrix(expected, predicted))

