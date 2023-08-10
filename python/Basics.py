import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = np.array([[3, -1.5,  2, -5.4], [0,  4,  -0.3, 2.1], [1,  3.3, 
-1.9, -4.3]])

data_standardized = preprocessing.scale(data)
print ("\nMean =", data_standardized.mean(axis=0))
print ("Std deviation =", data_standardized.std(axis=0))

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print ("\nMin max scaled data =", data_scaled)

data_normalized=preprocessing.normalize(data,norm='l1')
print("\nL1 normalized data =", data_normalized)

data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print ("\nBinarized data =", data_binarized)

encoder=preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print ("\nEncoded vector =", encoded_vector)

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
input_classes=['acm','ece','ieee','cse','it','csi']
label_encoder.fit(input_classes)
print("\nClass Mapping:")
for i, item in enumerate(label_encoder.classes_):
    print (item, '---->',i)

labels=['acm','ece','cse']
encoded_labels=label_encoder.transform(labels)
print("\nLabels=",labels)
print("Encoded Labels=", list(encoded_labels))

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print ("\nEncoded labels =", encoded_labels)
print ("Decoded labels =", list(decoded_labels))

import sys
import numpy as np

filename = "data_singlevar.txt"
X = []
y = []
with open(filename, "r") as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)


num_training = int(0.8 * len(X))
num_test = len(X) - num_training


# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])
# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])


from sklearn import linear_model
# Create linear regression object
linear_regressor = linear_model.LinearRegression()
# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

import matplotlib.pyplot as plt
y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()
y_test_pred = linear_regressor.predict(X_test)
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

import sklearn.metrics as sm
print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print ("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
print ("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
print ("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
print ("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

