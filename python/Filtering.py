import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import classification_report
import os.path
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier

df = pd.read_csv(open('sample.csv'))

def vectorize(df):
    vectorizer = TfidfVectorizer(stop_words="english",min_df=10,max_features=50)
    vectorizer.fit(df['title']+df['message'])
    pickle.dump(file=open("vector.pkl","w"),obj=vectorizer)

def cluster(df,number_of_clusters,batch_size,max_iter):
    vz = pickle.load(open('vector.pkl',"r"))
    data_points = len(df)
    class_val = []
    interia_list = []
    z = MiniBatchKMeans(n_clusters=number_of_clusters,batch_size=batch_size,max_iter=max_iter)
    for i in xrange(0,data_points,batch_size):
        features = vz.transform(df[i:i+batch_size]['title']+df[i:i+batch_size]['message'])
        z.partial_fit(features)
        interia_list.append(z.inertia_)
        class_val = class_val + list(z.labels_)
    new_df = df[0:data_points]
    new_df['class'] = class_val
    new_df.to_csv('ClusteredData.csv', index=False)

def classify(clusteredDataFile,test_size):
    vz = pickle.load(open('vector.pkl',"r"))
    clusteredData = pd.read_csv(clusteredDataFile)
    all_features = vz.transform(clusteredData['title']+clusteredData['message'])
    X_train, X_test, y_train, y_test = train_test_split(all_features,clusteredData['class'],test_size=test_size)
    clf = SGDClassifier()
    clf.fit(X_train,y_train)
    y_true,y_pred = y_test,clf.predict(X_test)
    pickle.dump(file=open("classifier.pkl","w"),obj=clf)

def get_classification_report(test_size):
    clusteredData = pd.read_csv('ClusteredData.csv')
    vz = pickle.load(open('vector.pkl',"r"))
    all_features = vz.transform(clusteredData['title']+clusteredData['message'])
    X_train, X_test, y_train, y_test = train_test_split(all_features,clusteredData['class'],test_size=test_size)
    clf = pickle.load(open("classifier.pkl","r"))
    report = classification_report(y_test,clf.predict(X_test), digits=10)
    print report

vectorize(df)

cluster(df,3,10,50)

classify('ClusteredData.csv',0.05)

get_classification_report(0.5)

