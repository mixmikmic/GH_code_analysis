from keras.datasets import reuters
from keras.preprocessing import sequence
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=20000)
X_train = sequence.pad_sequences(X_train, maxlen=1000)
X_test = sequence.pad_sequences(X_test, maxlen=1000)
y_train, y_test = to_categorical(y_train, 46), to_categorical(y_test, 46)

from keras.models import Sequential
from keras.layers import Masking, GlobalAveragePooling1D, Embedding, Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Masking(mask_value=0, input_shape=(1000,)))
model.add(Embedding(20000, 64, input_length=1000))
model.add(GlobalAveragePooling1D())

model.add(Dense(500, activation='relu'))
model.add(Dense(46, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=128, verbose=2)

import keras.backend as K
import numpy as np

get_conv_outputs = K.function([model.layers[0].input, K.learning_phase()], [model.layers[3].output])

x_repr = [get_conv_outputs([[X_test[i]], 0])[0] for i in range(len(X_test))]
x_repr = np.float64(x_repr).reshape((X_test.shape[0], 500))
labels = np.argmax(y_test, axis=1)
print(x_repr.shape)
print(labels.shape)

# Perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_repr = pca.fit_transform(x_repr)

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(1,1, figsize=(16,16))

# make the scatter
scat = ax.scatter(pca_repr[:, 0], pca_repr[:,1], c=labels, s=20, cmap=plt.cm.jet)
cb = plt.colorbar(scat, spacing='proportional', ticks=np.linspace(0,46,47))
ax.set_title('Visualization using PCA')
plt.show()

# Perform t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne_repr = tsne.fit_transform(x_repr)

fig, ax = plt.subplots(1,1, figsize=(16,16))

# make the scatter
scat = ax.scatter(tsne_repr[:, 0], tsne_repr[:,1], c=labels, s=20, cmap=plt.cm.jet)
cb = plt.colorbar(scat, spacing='proportional', ticks=np.linspace(0,46,47))
ax.set_title('Visualization using t-SNE')
plt.show()

