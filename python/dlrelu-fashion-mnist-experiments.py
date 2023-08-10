from tensorflow.examples.tutorials.mnist import input_data

fashion = input_data.read_data_sets('/home/darth/GitHub Projects/fashion-mnist/data/fashion', one_hot=True)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

pca = PCA(n_components=784)
pca.fit(fashion.train.images)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(fashion.train.images)

normalized_train = scaler.transform(fashion.train.images)
normalized_test = scaler.transform(fashion.test.images)

pca = PCA(n_components=256)

reduced_train = pca.fit_transform(normalized_train)

reduced_test = pca.transform(normalized_test)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

from models.cnn_keras import CNN

model = CNN(activation='relu',
            classifier='relu',
            input_shape=(16, 16, 1),
            loss='categorical_crossentropy',
            num_classes=fashion.train.labels.shape[1],
            optimizer='adam',
            return_summary=True)

model.train(batch_size=256,
            n_splits=10,
            validation_split=0.,
            verbose=0,
            train_features=np.reshape(reduced_train, (-1, 16, 16, 1)),
            train_labels=fashion.train.labels,
            epochs=32)

report, confusion_matrix = model.evaluate(batch_size=256,
                                          test_features=np.reshape(reduced_test, (-1, 16, 16, 1)),
                                          test_labels=fashion.test.labels,
                                          class_names=class_names)

print(report)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, annot_kws={'size': 12}, fmt='.2f')

from models.dnn_keras import DNN

model = DNN(activation='relu',
            classifier='relu',
            dropout_rate=0.1,
            loss='categorical_crossentropy',
            optimizer='adam',
            num_classes=fashion.train.labels.shape[1],
            num_features=reduced_train.shape[1],
            num_neurons=[512, 512, 512],
            return_summary=True)

model.train(batch_size=256,
            n_splits=10,
            epochs=32,
            validation_split=0.,
            verbose=0,
            train_features=np.reshape(reduced_train, (-1, 256)),
            train_labels=fashion.train.labels)

report, confusion_matrix = model.evaluate(batch_size=256,
                                          test_features=np.reshape(reduced_test, (-1, 256)),
                                          test_labels=fashion.test.labels,
                                          class_names=class_names)

print(report)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, annot_kws={'size': 12}, fmt='.2f')

from models.cnn_keras import CNN

model = CNN(activation='relu',
            classifier='softmax',
            input_shape=(16, 16, 1),
            loss='categorical_crossentropy',
            num_classes=fashion.train.labels.shape[1],
            optimizer='adam',
            return_summary=True)

model.train(batch_size=256,
            n_splits=10,
            validation_split=0.,
            verbose=0,
            train_features=np.reshape(reduced_train, (-1, 16, 16, 1)),
            train_labels=fashion.train.labels,
            epochs=32)

report, confusion_matrix = model.evaluate(batch_size=256,
                                          test_features=np.reshape(reduced_test, (-1, 16, 16, 1)),
                                          test_labels=fashion.test.labels,
                                          class_names=class_names)

print(report)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, annot_kws={'size': 12}, fmt='.2f')

from models.dnn_keras import DNN

model = DNN(activation='relu',
            classifier='softmax',
            dropout_rate=0.1,
            loss='categorical_crossentropy',
            optimizer='adam',
            num_classes=fashion.train.labels.shape[1],
            num_features=reduced_train.shape[1],
            num_neurons=[512, 512, 512],
            return_summary=True)

model.train(batch_size=256,
            n_splits=10,
            epochs=32,
            validation_split=0.,
            verbose=0,
            train_features=np.reshape(reduced_train, (-1, 256)),
            train_labels=fashion.train.labels)

report, confusion_matrix = model.evaluate(batch_size=256,
                                          test_features=np.reshape(reduced_test, (-1, 256)),
                                          test_labels=fashion.test.labels,
                                          class_names=class_names)

print(report)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, annot_kws={'size': 12}, fmt='.2f')

