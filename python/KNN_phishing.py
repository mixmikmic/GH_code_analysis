import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import cross_val_score

# Importing Data from arff File using scipy arff module
phishing_data_raw = loadarff("PhishingData.arff")
phishing_data_raw

# Selection of the Data and converting data into numpy format for flexibility in cleaning
phising_data_array = np.array(phishing_data_raw[0])
phising_data_array

# Converting the numpy array into Pandas data frame  and casting the coloumns to numeric type
phising_data_frame = pd.DataFrame(phising_data_array).apply(pd.to_numeric)
phising_data_frame[0:10]

col_names = list(phising_data_frame)# Getting Column names of the pandas data frame
print("Column_names:" + ", ".join(col_names))
# The scikit learn package takes data in form of labels and predictors
# Extracting Result coloum to pass as target coloum
labels = phising_data_frame["Result"].values
print(labels)

predictor_cols = col_names[0:len(col_names)-1]# selecting the predictor column names
print("Predictor_Columns:" + ", ".join(predictor_cols))
#extracting Predictor points 
predictors = phising_data_frame[predictor_cols].values
predictors

# Defining The parameters in KNN Classifier
clf_knn = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance'
    )
clf_knn = clf_knn.fit(predictors,labels)

# Computing the Cross validation score with 5-fold cross validation
score_knn = cross_val_score(clf_knn, predictors, labels, cv=5)
print("Cross Validation score : " + str(score_knn))
print("Cross Validation Mean score : " + str(score_knn.mean()))



