from keras.utils import to_categorical
from sklearn.datasets import load_breast_cancer

features = load_breast_cancer().data
labels = load_breast_cancer().target

# one-hot encode the labels
labels = to_categorical(labels)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.30, stratify=labels)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_features)

normalized_train = scaler.transform(train_features)
normalized_test = scaler.transform(test_features)

class_names = ['Malignant', 'Benign']

from models.dnn_keras import DNN

model = DNN(activation='relu',
            classifier='relu',
            dropout_rate=0.1,
            loss='categorical_crossentropy',
            optimizer='adam',
            num_classes=train_labels.shape[1],
            num_features=train_features.shape[1],
            num_neurons=[128, 64, 32],
            return_summary=True)

model.train(batch_size=256,
            n_splits=10,
            epochs=32,
            validation_split=0.,
            verbose=0,
            train_features=train_features,
            train_labels=train_labels)

report, confusion_matrix = model.evaluate(batch_size=256,
                                          test_features=test_features,
                                          test_labels=test_labels,
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
            num_classes=train_labels.shape[1],
            num_features=train_features.shape[1],
            num_neurons=[128, 64, 32],
            return_summary=True)

model.train(batch_size=256,
            n_splits=10,
            epochs=32,
            validation_split=0.,
            verbose=0,
            train_features=train_features,
            train_labels=train_labels)

report, confusion_matrix = model.evaluate(batch_size=256,
                                          test_features=test_features,
                                          test_labels=test_labels,
                                          class_names=class_names)

print(report)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, annot=True, annot_kws={'size': 12}, fmt='.2f')

