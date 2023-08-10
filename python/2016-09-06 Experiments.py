import sys
sys.path.append('../src/mane/prototype/')
import numpy as np
import graph as g
import pickle as p

from sklearn.preprocessing import normalize, scale, MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

# Load weight
with open('../src/mane/prototype/embeddings/BC3_deepwalk.weights', 'rb') as f:
    w = p.load(f)
# Load graph
bc = g.graph_from_pickle('../src/mane/data/blogcatalog3.graph', 
                         '../src/mane/data/blogcatalog3.community')

emb = (w[0] + w[1] )/2
emb = normalize(emb)

x_train, yl_train, x_test, yl_test = bc.get_ids_labels(0.5)

lg = OneVsRestClassifier(LogisticRegression(C=2), n_jobs=-1)

X_train = [emb[i] for i in x_train]
Y_train = MultiLabelBinarizer(classes=range(0,39)).fit_transform(yl_train)

lg.fit(X_train, Y_train)

X_test = [emb[i] for i in x_test]
Y_test = MultiLabelBinarizer(classes=range(0,39)).fit_transform(yl_test)

pred = lg.predict_proba([i for i in X_test])
num_pred = []
for i, j in enumerate(pred):
    k = len(yl_test[i])
    num_pred.append(j.argsort()[-k:])

Y_pred = MultiLabelBinarizer(classes=range(0,39)).fit_transform(num_pred)

f1_score(y_pred=Y_pred, y_true=Y_test, average='weighted')

len(x_train)

x_train[4]

yl_train[4]

x_test[5]

yl_test[5]

num_pred[5]



