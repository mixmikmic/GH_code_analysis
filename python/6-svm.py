get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

def logit_loss(s):
    return np.log(1 + np.exp(-s))

def hinge_loss(s):
    return np.maximum(0, 1-s)

s = np.arange(-3, 3, 0.1)

# plot the sigmoid curve.
plt.figure(figsize=(6, 3))
plt.plot(s, logit_loss(s), label="Logistic")
plt.plot(s, hinge_loss(s), color='r', label="Hinge loss")
plt.grid()
plt.legend()
plt.show()

np.random.seed(1)

def svm(x, y, x_test, y_test, num_iteration=1000, learning_rate=1e-2, regularization=0.1):
    r, c = x.shape

    p = c + 1
    X = np.hstack((np.ones((r,1)), x))
    beta = 2*np.random.randn(p, 1)-1
    
    X_test = np.hstack((np.ones((x_test.shape[0],1)), x_test))
    acc, acc_test = [], []
    
    for i in xrange(num_iteration):
        score = X.dot(beta)
        db = (score * y) < 1
        dbeta = X.T.dot(db*y) / r;
        beta = beta + learning_rate *dbeta - regularization*beta; 
        
        # accuracy on training and testing data
        acc.append(np.mean(np.sign(score) == y))
        acc_test.append(np.mean(np.sign(X_test.dot(beta)) == y_test))
    
    return beta, acc, acc_test

from common import load_digits, split_samples

# Load digits and split the data into two sets.
digits, labels = load_digits(subset=[3, 5], normalize=True)
training_digits, training_labels, testing_digits, testing_labels = split_samples(digits, labels)

print '# training:', training_digits.shape[0]
print '# testing :', testing_digits.shape[0]

# Transform labels from {0, 1} to {-1, 1}.
training_labels = 2* training_labels -1
testing_labels = 2* testing_labels -1

# Train a svm classifier.
beta, acc, acc_test = svm(training_digits, training_labels, testing_digits, testing_labels)

# Plot the accuracy.
plt.figure()
plt.plot(range(len(acc)), acc, color='b', label='Training Accuracy')
plt.plot(range(len(acc)), acc_test, color='r', label='Testing Accuracy')
plt.legend(loc='best')
plt.show()



