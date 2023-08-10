import h5py
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np

f1 = h5py.File('train.h5', 'r')
list(f1.items())

f2 = h5py.File('test.h5', 'r')
list(f2.items())

print(list(f1.keys()))
print(list(f2.keys()))

train_data = f1['data'][()]
train_labels = f1['label'][()]
test_data = f2['data'][()]
test_labels = f2['label'][()]
print("Train data shape: {}".format(train_data.shape))
print("Train labels shape: {}".format(train_labels.shape))
print("Test data shape: {}".format(test_data.shape))
print("Test labels shape: {}".format(test_labels.shape))

# Convert labels to one hot encoding
encoder = OneHotEncoder(sparse=False)
train_labels = train_labels.reshape(-1, 1)
print(train_labels.shape)
test_labels = test_labels.reshape(-1, 1)
print(test_labels.shape)
train_labels_one_hot = encoder.fit_transform(train_labels)
test_labels_one_hot = encoder.fit_transform(test_labels)
print("Train labels shape: {}".format(train_labels_one_hot.shape))
print("Test labels shape: {}".format(test_labels_one_hot.shape))

pickle.dump(train_data, open("CIFAR_10_train_data.pkl", 'wb'), protocol=2)
pickle.dump(train_labels_one_hot, open("CIFAR_10_train_labels.pkl", 'wb'), protocol=2)
pickle.dump(test_data, open("CIFAR_10_test_data.pkl", 'wb'), protocol=2)
pickle.dump(test_labels_one_hot, open("CIFAR_10_test_labels.pkl", 'wb'), protocol=2)

# Create subset of data with only two classes (airplane or cat)
airplane = train_labels==0
cat = train_labels==3
airandcat_train = (train_labels==0) | (train_labels==3)
print(airplane[:10])
print(cat[:10])
print(airandcat_train[:10])
airandcat_train_ix = np.where(airandcat_train)[0]
print(airandcat_train_ix[:10])
airandcat_test = (test_labels==0) | (test_labels==3)
print(airandcat_test[:10])
airandcat_test_ix = np.where(airandcat_test)[0]
print(airandcat_test_ix[:10])

train_data_2classes = train_data[airandcat_train_ix,::]
train_labels_2classes = train_labels[airandcat_train_ix]
test_data_2classes = test_data[airandcat_test_ix,::]
test_labels_2classes = test_labels[airandcat_test_ix]

print(sum(airandcat_train))
print(airandcat_train.shape)
print(train_data.shape)
print(train_data_2classes.shape)
print(train_labels_2classes.shape)
print(sum(airandcat_test))
print(airandcat_test.shape)
print(test_data.shape)
print(test_data_2classes.shape)
print(test_labels_2classes.shape)

# Convert labels to one hot encoding
encoder = OneHotEncoder(sparse=False)
train_labels_2classes = train_labels_2classes.reshape(-1, 1)
print(train_labels_2classes.shape)
test_labels_2classes = test_labels_2classes.reshape(-1, 1)
print(test_labels_2classes.shape)
train_labels_2classes_one_hot = encoder.fit_transform(train_labels_2classes)
test_labels_2classes_one_hot = encoder.fit_transform(test_labels_2classes)
print("Train 2 classes labels shape: {}".format(train_labels_2classes_one_hot.shape))
print("Test 2 classes labels shape: {}".format(test_labels_2classes_one_hot.shape))

pickle.dump(train_data_2classes, open("CIFAR_2_train_data.pkl", 'wb'), protocol=2)
pickle.dump(train_labels_2classes_one_hot, open("CIFAR_2_train_labels.pkl", 'wb'), protocol=2)
pickle.dump(test_data_2classes, open("CIFAR_2_test_data.pkl", 'wb'), protocol=2)
pickle.dump(test_labels_2classes_one_hot, open("CIFAR_2_test_labels.pkl", 'wb'), protocol=2)

