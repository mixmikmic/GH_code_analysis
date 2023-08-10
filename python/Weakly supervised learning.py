import numpy as np
import keras.backend as K

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, n_features=2, centers=[(0, 0), (0, 2)], random_state=1)
y = y.astype(float)

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

# make 2 batches of data at random
indices = np.random.permutation(len(X))
batch1 = indices[:400]
batch2 = indices[400:]
y_props = np.zeros(len(y))
y_props[batch1] = np.mean(y[batch1])
y_props[batch2] = np.mean(y[batch2])

y_props[:10]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, y_props_train, y_props_test = train_test_split(X, y, y_props, test_size=0.25)

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

x = Input(shape=(2,))
h = Dense(20, activation="relu")(x)
h = Dense(20, activation="relu")(h)
h = Dense(1, activation="sigmoid")(h)
net1 = Model(x, h)
net1.compile(loss="binary_crossentropy", optimizer=Adam())

x = Input(shape=(2,))
h = Dense(20, activation="relu")(x)
h = Dense(20, activation="relu")(h)
h = Dense(1, activation="sigmoid")(h)
net2 = Model(x, h)
def loss_function(ytrue, ypred):
    # Assuming that ypred contains the same ratio replicated
    #loss = K.sum(ypred)/ypred.shape[0] - K.sum(ytrue)/ypred.shape[0]   # does not work with tensorflow backend
    den = K.cast(K.shape(ypred)[0], dtype="float32")
    loss = K.sum(ypred) / den - K.sum(ytrue) / den
    #loss = K.mean(ypred) - K.mean(ytrue)  # equivalent to above
    loss = K.square(loss)
    return loss
net2.compile(loss=loss_function, optimizer=Adam())

net1.fit(X_train, y_train, nb_epoch=50, validation_data=(X_test, y_test))

net2.fit(X_train, y_props_train, nb_epoch=50, validation_data=(X_test, y_props_test))

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, net1.predict(X_test)))
print(roc_auc_score(y_test, net2.predict(X_test)))

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, net1.predict(X_test))
plt.plot(fpr, tpr, label="full")
fpr, tpr, _ = roc_curve(y_test, net2.predict(X_test))
plt.plot(fpr, tpr, label="weakly")

plt.legend()
plt.show()

