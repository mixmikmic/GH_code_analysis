import pandas as pd

reviews = pd.read_csv('../data/en_reviews.csv', sep='\t', header=None, names =['rating', 'text'])
reviews[35:45]

target = reviews['rating']
data = reviews['text']
names = ['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5']

print(data[:5])
print(target[:5])

from nltk.tokenize.casual import casual_tokenize
tokens = data.apply(lambda x: casual_tokenize(x))

import numpy as np
DIM = 300 #dimension of the word2vec vectors

word_vectors = {}

with open('../data/crawl-300.vec') as f:
    f.readline()
    for line in f.readlines():
        items = line.strip().split()
        word_vectors[items[0]] = np.array(items[1:], dtype=np.float32)

vectors = []
for line in tokens:
    vec_list = []
    for token in line:
        if token in word_vectors.keys():
            vec_list.append(word_vectors.get(token))
    if len(vec_list) == 0:
        vec_list.append(np.zeros(DIM))        
    vectors.append(np.average(vec_list, axis=0))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(vectors, target, test_size=0.2)
print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf_pipeline = Pipeline([('std', StandardScaler()),
                         ('svm', SVC(kernel='rbf'))])
    
clf_pipeline.fit(X_train, y_train)

y_pred = clf_pipeline.predict(X_test)

from sklearn import metrics

print()
print("ML MODEL REPORT")
print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred,
                                            target_names=names))

