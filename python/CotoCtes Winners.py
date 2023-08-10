from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from time import time


#filename = r'C:\P\Machine Learning\cotoctes\ucici.csv'
filename = r'C:\P\Machine Learning\cotoctes\ucici_full.csv'
import pandas as pd
#training = pd.read_csv(filename,delimiter='|',header=None,names = ['cat', 'data'])
training = pd.read_csv(filename,delimiter='|',header=None,names = ['cat','other', 'data'])
training['yr'] = training['other'].str.slice(1, 3)
training = training[training['yr'].isin(['15','16','17'])]
X_train,X_test = training[0:-5000],training[-5000:]

print(X_train.shape,X_test.shape)
for a, b in zip(X_train.data[0:2], X_train.cat[0:2]):
    print("{0} => {1}".format(a,b))
    
#def size_mb(docs):
#    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

#data_train_size_mb = size_mb(X_train.data)
#data_test_size_mb = size_mb(X_test.data)

#print("%d items - %0.3fMB (training set)" % (
#    len(X_train.data), data_train_size_mb))
#print("%d items - %0.3fMB (test set)" % (
#    len(X_test.data), data_test_size_mb))
print(X_train.head(5))

parameters = {
    'solver': [('adam')],
    'alpha': (0.001, 0.00001, 0.000001),
    'hidden_layer_sizes': [(4)],
}

from sklearn.model_selection import ParameterGrid
for p in list(ParameterGrid(parameters)):
    t0 = time()
    print(p)
    pipeline = Pipeline([
        ('vect', HashingVectorizer()),
        ('clf', MLPClassifier(**p)),
        ])
    pipeline.fit(X_train.data, X_train.cat)
    print("Testing {0:.2f} / Training {1:.2f}".format(np.mean(pipeline.predict(X_train.data) == X_train.cat),np.mean(pipeline.predict(X_test.data) == X_test.cat)))
    print("done in %0.3fs" % (time() - t0))

filename = r'C:\P\Machine Learning\cotoctes\testovaci.csv'
testing = pd.read_csv(filename,delimiter='|',header=None,names = ['data'])
#with open(r'C:\P\Machine Learning\cotoctes\outputBestMPL.csv','w') as f:
#    wr = csv.writer(f, delimiter='\n')
#    wr.writerow(gs_clf.predict(testing.data))

from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Perceptron
#from sknn.mlp import MultiLayerPerceptron
pipeline2 = Pipeline([
        ('vect',Pipeline([
                   ('union', FeatureUnion(
                           transformer_list=[
                       ('words', CountVectorizer(analyzer='word',ngram_range=(1,2))),
                       ('chars', CountVectorizer(analyzer='char',ngram_range=(1,3))),
                        ],)),
                   ('tfidf', TfidfTransformer(use_idf=False)),   
                    ])),
        ('clf', Perceptron(max_iter=25)),
        #('clf', MultiLayerPerceptron(max_iter=50,dropout_rate=0.5)),
        ])
pipeline2.fit(X_train.data, X_train.cat)
print("Testing {0:.2f} / Training {1:.2f}".format(np.mean(pipeline2.predict(X_train.data) == X_train.cat),np.mean(pipeline2.predict(X_test.data) == X_test.cat)))

import csv
with open(r'C:\P\Machine Learning\cotoctes\outputCountPercFull.csv','w') as f:
    wr = csv.writer(f, delimiter='\n')
    wr.writerow(pipeline2.predict(testing.data))

pipeline2.predict(['Pioneer má první interní mechaniku pro přehrávání Ultra HD disků na PC'])

testing['cat'] = pipeline2.predict(testing.data)
print(testing.head(10))
print(testing.tail(10))



