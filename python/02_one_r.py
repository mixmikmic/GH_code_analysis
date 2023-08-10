from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset.DESCR)

X = dataset.data
y = dataset.target
n_samples, n_features = X.shape

# Compute the mean for each attributes
attribute_means = X.mean(axis=0)
assert attribute_means.shape == (n_features,)

X_d = np.array(X >= attribute_means, dtype='int')

# Split into training and test set
from sklearn.model_selection import train_test_split

# Seed our random state so that we will get reproducible results
random_state = 14

X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)

print("There are {} training samples".format(y_train.shape))
print("There are {} test samples".format(y_test.shape))

from collections import defaultdict
from operator import itemgetter

def train_feature_value(X, y_true, feature_index, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_count.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    incorrect_predictions = [class_count for class_value, class_count in class_counts.items() if class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class, error


