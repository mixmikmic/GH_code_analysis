get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

def logit_loss(s):
    return np.log(1 + np.exp(-s))

def hinge_loss(s):
    return np.maximum(0, 1-s)

def exp_loss(s):
    return np.exp(-s)

s = np.arange(-1, 5, 0.1)

# plot the sigmoid curve.
plt.figure(figsize=(6, 3))
plt.plot(s, logit_loss(s), label="Logistic")
plt.plot(s, hinge_loss(s), label="Hinge loss")
plt.plot(s, exp_loss(s), label="Exp loss")
plt.grid()
plt.legend()
plt.show()

np.random.seed(1)

def adaboost(X, y, X_test, y_test, num_iteration=50):
    n, p = X.shape
    
    th = 0.8
    X_transformed = 2*(X > th) - 1
    X_test_transformed = 2*(X_test > th) - 1
    
    acc, acc_test = [], []
    
    beta = np.zeros((p, 1))
    
    # Initially, all weights are same.
    weights = np.ones((n, 1))
    
    # Results from weak classifier: yi * h(x_i)
    weak_results = y*X_transformed > 0
    
    for i in xrange(num_iteration):
        # Normalize weights.
        weights = weights / np.sum(weights)
        
        # Calculate the error on weighted examples.
        weighted_weak_results = weights * weak_results
        weighted_accuracy = np.sum(weighted_weak_results, axis=0)
        error = 1 - weighted_accuracy

        # Select the one with the min error and update the beta.
        j = np.argmin(error)
        dbeta = np.log((1 - error[j])/error[j])/2
        beta[j] = beta[j] + dbeta
        
        # Update the weights sequentially to avoid overflow.
        weights = weights * np.exp(-dbeta*weak_results[:,j].reshape((n, 1)))
        
        # Calcualte accuracy on training and testing data.
        score = X_transformed.dot(beta)
        acc.append(np.mean(np.sign(score) == y))
        acc_test.append(np.mean(np.sign(X_test_transformed.dot(beta)) == y_test))
    
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
beta, acc, acc_test = adaboost(training_digits, training_labels, testing_digits, testing_labels)

# Plot the accuracy.
plt.figure()
plt.plot(range(len(acc)), acc, color='b', label='Training Accuracy')
plt.plot(range(len(acc)), acc_test, color='r', label='Testing Accuracy')
plt.legend(loc='best')
plt.show()



