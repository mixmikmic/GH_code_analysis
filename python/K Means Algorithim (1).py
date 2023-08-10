#Importing Libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from  sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist

#Import CSV data

import sys
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_8681a3baf2ff4fdc8fc7cb341e6faff6 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='2B1RbhJ9pdEcoF21dlXb9fukDuVV_p5e_Z475-ZFcJnO',
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_8681a3baf2ff4fdc8fc7cb341e6faff6.get_object(Bucket='kmeansclusteringwithdnnb7d592a69aa14a2fbdca2a069a215a34',Key='a1Finalok_2.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_2 = pd.read_csv(body, encoding = 'latin1')
df_data_2.head()


#Entering default for NULL entries

NewData = df_data_2.iloc[:,:13].values

# Binary Encoder

dataset = df_data_2
dataset.drop('Source_IP',axis=1, inplace=True)
dataset.drop('Year',axis=1, inplace=True)
X = dataset.iloc[:,0:10].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le_x = LabelEncoder()
X[:,0] = le_x.fit_transform(X[:,0])

lex_x1 = LabelEncoder()
X[:,1] = lex_x1.fit_transform(X[:,1])

lex_x2 = LabelEncoder()
X[:,2] = lex_x2.fit_transform(X[:,2])

lex_x3 = LabelEncoder()
X[:,8] = lex_x3.fit_transform(X[:,8])

onehotencoder = OneHotEncoder(categorical_features="all")
X = onehotencoder.fit_transform(X).toarray()

#Kmeans for hyperparameter selection

Kr = [2, 4, 6, 8, 10, 12, 16, 18, 24, 28, 32]
distortions = []

for k in Kr:
    print("Clusters: "+str(k))
    kmeans_model = KMeans(n_clusters= k, init = 'k-means++', n_init = 10, max_iter = 500).fit(X)
    label = kmeans_model.labels_
    distortions.append(sum(np.min(cdist(X, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    print("The average silhouette_score is :"+str(sum(np.min(cdist(X, kmeans_model.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]))

#Plot the data and try to find a Elbow in the graph 
plt.plot(distortions)

# Using optimum Number of Clusters = 8

kmeans_model = KMeans(n_clusters= 8, init = 'k-means++', n_init = 20, max_iter = 2000).fit(X)

# Output Parameters of K-Mean

labels = kmeans_model.labels_
cluster_centers = kmeans_model.cluster_centers_
squared_distance = kmeans_model.inertia_
Prediction = kmeans_model.predict(X)

Prediction = np.reshape(Prediction, (-1,1))
print("Labels : "+str(labels))
print("Labels Shape: "+str(labels.shape))
print("Cluster Points : "+str(cluster_centers))
print("Squared Distance : "+str(squared_distance))
print("Output Shape"+str(Prediction.shape))

# Visualize the Data

pca = PCA(n_components = 2)
X_New = pca.fit_transform(X)

final = np.concatenate((X_New, Prediction), axis = 1)
print("Final Shape: "+str(final.shape))


plt.figure()

for i in range(16):
    indexes = final[:,2] == i
    plt.scatter(final[indexes, 0], final[indexes, 1], s = 2)

plt.show()

print("Retention of variance:")
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

# Output the clustered visualised data and perform manual analysis to determine anomalies and patterns in the data

for i in range(8):
    indexes = final[:,2] == i
    # df_data_2[indexes] 8 outputs

# Concatenating Kmeans Predictions with Input Data to create input Data for RNN Classification

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation

# Creating Train_x and Train_Y

Train_x = X
Train_y = Prediction

print("Train X"+str(Train_x.shape))
print("Train Y"+str(Train_y.shape))

# Building function model
def build_model(layers = [231, 8, 100, 50]):
    
    model = Sequential()
    
    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],input_shape=Train_x.shape,return_sequences=True))
    
    model.add(LSTM(layers[2], return_sequences=False))
    
    model.add(Dense(output_dim=layers[3]))
    
    model.add(Activation("softmax"))
    model.compile(loss="mse", optimizer="adam")
    
    return model

model = build_model()

model.fit(Train_x,Train_y, batch_size=64, nb_epoch=5000, validation_split=0.05)



