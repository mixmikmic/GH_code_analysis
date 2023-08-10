get_ipython().magic('pylab inline')

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import classification_report, confusion_matrix

n_samples = 10000
n_classes = 3
n_features = 2

# centers - number of classes
# n_features - dimension of the data
X, y_int = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features,     cluster_std=0.5, random_state=0)

# No need to convert the features and targets to the 32-bit format as in plain theano.

# labels need to be one-hot encoded (binary vector of size N for N classes)
y = np_utils.to_categorical(y_int, n_classes)

# visualize the data for better understanding
def plot_2d_blobs(dataset):
    X, y = dataset
    axis('equal')
    scatter(X[:, 0], X[:, 1], c=y, alpha=0.1, edgecolors='none')

plot_2d_blobs((X, y_int))

# split the data into training, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# the model is just a sequence of transformations - layer weights, activations, etc.
model = Sequential()
# weights from input to hidden layer - linear transform
model.add(Dense(3, input_dim=n_features))
# basic non-linearity
model.add(Activation("tanh"))
# weights from hidden to output layer
model.add(Dense(n_classes))
# nonlinearity suitable for a classifier
model.add(Activation("softmax"))

# - loss function suitable for multi-class classification
# - plain stochastic gradient descent with mini-batches
model.compile(loss='categorical_crossentropy', optimizer='sgd')

model.fit(X_train, y_train, nb_epoch=5, batch_size=32);

def evaluate_accuracy(X, y, label):
    _, accuracy = model.evaluate(X_train, y_train, show_accuracy=True)
    print('training accuracy:', 100 * accuracy, '%')

evaluate_accuracy(X_train, X_train, 'training')
evaluate_accuracy(X_test, X_test, 'test')

y_test_pred = model.predict_classes(X_test)

plot_2d_blobs((X_test, y_test_pred))

