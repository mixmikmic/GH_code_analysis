import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

from neural_network import NeuralNetworkImpl
from utils import compute_accuracy_multilabel

# Generate train and test datasets, as plotted below.
np.random.seed(12)
num_samples = 10000

negative_points1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], int(num_samples/2))
negative_points2 = np.random.multivariate_normal([2, 7], [[1, .75],[.75, 1]], int(num_samples/2))
positive_points = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_samples)

all_points = np.vstack((negative_points1[:int(num_samples/4)],
    negative_points2[:int(num_samples/4)],
    positive_points[:int(num_samples/2)])).astype(np.float32)
point_labels = np.hstack((np.repeat([[1,0,0]], int(num_samples/4), axis=0).T,
                          np.repeat([[0,1,0]], int(num_samples/4), axis=0).T,
                          np.repeat([[0,0,1]], int(num_samples/2), axis=0).T))

test_points = np.vstack((negative_points1[int(num_samples/4):],
    negative_points2[int(num_samples/4):],
    positive_points[int(num_samples/2):])).astype(np.float32)
test_labels = np.hstack((np.repeat([[1,0,0]], int(num_samples/4), axis=0).T,
                         np.repeat([[0,1,0]], int(num_samples/4), axis=0).T,
                         np.repeat([[0,0,1]], int(num_samples/2), axis=0).T))

plt.figure(figsize=(12,8))
plt.scatter(all_points[:, 0], all_points[:, 1], c = np.argmax(point_labels, axis=0), alpha = .4)
plt.show()

model_nn = NeuralNetworkImpl(layer_sizes=[10, 5, 3], layer_activations=['relu', 'relu', 'softmax'],
    alpha=0.01, epochs=500, mini_batch_size=32, regularization=0.1, optimization_algorithm='adam')
model_nn.train(all_points.T, point_labels)
accuracy = compute_accuracy_multilabel(model_nn, test_points.T, test_labels)
print("Neural net model test accuracy: {}".format(accuracy))

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(all_points, point_labels.T, epochs=500, batch_size=32, verbose=0)
score = model.evaluate(test_points, test_labels.T, batch_size=32)
print("\nKeras NN loss/accuracy:")
for label, value in zip(model.metrics_names, score):
    print("{}: {}".format(label, value))



