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
with open('../src/mane/prototype/embeddings/BC3047.weights', 'rb') as f:
    w = p.load(f)
# Load graph
bc = g.graph_from_pickle('../src/mane/data/blogcatalog3.graph', 
                         '../src/mane/data/blogcatalog3.community')

emb = (w[0] + w[1]) / 2
emb = normalize(emb)

(w[0] + w[1])[0]

normalize(_)

emb[0]

x_train, yl_train, x_test, yl_test = bc.get_ids_labels(0.5)

X_train = [emb[i] for i in x_train]
Y_train = MultiLabelBinarizer().fit_transform(yl_train)

Y_train.shape

for i,j in bc._communities.items():
    if 39 in j:
        print(i)

bc._communities[1465]

lg = OneVsRestClassifier(LogisticRegression(C=1e5))

lg.fit(X_train, Y_train)

lg.predict(emb[9566].reshape(1,-1))

emb[5].dot(emb[0])

x_train[0]

x_train[1]

Y_train[8]

lg.predict_proba(emb[1234].reshape(1,-1))

bc._communities[1234]

lg.predict_proba(emb[1234].reshape(1,-1)).argsort()[0][-4:]

lg.predict_proba(emb[5437].reshape(1,-1)).argsort()[0]

bc._communities[5437]

for i in bc[5437]:
    print(bc._communities[i])

bc[5437]

for i in bc[7999]:
    if 32 in bc._communities[i]:
        print(i)

lg.predict_proba(emb[6984].reshape(1,-1))[0].argmax()

for x in [14,691,1250,1344,1465,1550,4709,7759]:
    if x in x_train:
        print('la')



