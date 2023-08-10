import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 2')

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

mnist = fetch_mldata('MNIST original')
X_train, X_test = mnist.data[:60000] / 255., mnist.data[60000:] / 255.
y_train, y_test = mnist.target[:60000], mnist.target[60000:]

X_train, y_train = shuffle(X_train, y_train)

X_train.shape, X_test.shape

from sklearn.ensemble import RandomForestClassifier
from helpers import Timer

rf = RandomForestClassifier(n_estimators=100)

with Timer():
    rf.fit(X_train, y_train)
rf.score(X_test, y_test)

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(n_iter=10, random_state=0)
with Timer():
    sgd.fit(X_train, y_train)
    
sgd.score(X_test, y_test)

X_small, y_small = X_train[::100], y_train[::100]
print(X_small.shape)

rf = RandomForestClassifier(n_estimators=100)

with Timer():
    rf.fit(X_small, y_small)
rf.score(X_test, y_test)

rf.apply(X_small).shape

rf.apply(X_small)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder().fit(rf.apply(X_small))
X_train_transformed = ohe.transform(rf.apply(X_train))
X_train_transformed

sgd = SGDClassifier(n_iter=10, random_state=0)
with Timer():
    sgd.fit(X_train_transformed, y_train)
sgd.score(ohe.transform(rf.apply(X_test)), y_test)

from sklearn.utils import gen_batches
sgd = SGDClassifier(random_state=0)

for i in range(10):
    for batch in gen_batches(len(X_train), batch_size=1000):
        X_batch = X_train[batch]
        y_batch = y_train[batch]
        X_batch_transformed = ohe.transform(rf.apply(X_batch))
        sgd.partial_fit(X_batch_transformed, y_batch, classes=range(10))
sgd.score(ohe.transform(rf.apply(X_test)), y_test)



