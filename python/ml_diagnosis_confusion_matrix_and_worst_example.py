import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn import naive_bayes

# let's plot inline
get_ipython().magic('matplotlib inline')

# load the digits dataset
from sklearn.datasets import load_digits
digits = load_digits()

# install watermark using
# %install_ext https://raw.githubusercontent.com/rasbt/watermark/master/watermark.py
get_ipython().magic('load_ext watermark')
# show a watermark for this environment
get_ipython().magic('watermark -d -m -v -p numpy,matplotlib -g')

# start a separate QTconsole for interactive coding
#%qtconsole

# flatten and binarise the 2D input image - this is our basic feature set
# we could improve the features here by adding new columns to X
arr_digits_greyscale = [digit.flatten() for digit in digits.images]
arr_digits = [digit.flatten()>=8 for digit in digits.images]
X_greyscale = np.vstack(arr_digits_greyscale)
X = np.vstack(arr_digits)
y = digits.target
print("X has shape {}, y has shape {}".format(X.shape, y.shape))

def show_example(idx, X, y):
    """Show an example as an image from X"""
    print("y (target label) = ", y[idx])
    print("Image for X[{}] = ".format(idx))
    plt.matshow(X[idx].reshape((8,8)))
    
show_example(0, X, y)

show_example(11, X, y)

clf = DummyClassifier()
cv = cross_validation.StratifiedKFold(y=y, n_folds=10, shuffle=True)

scores = cross_validation.cross_val_score(clf, X, y, cv=cv)

"Scores: {:0.3f} +/- {:0.3f} (2 s.d.)".format(scores.mean(), scores.std()*2)
# We expect a 10% score if we're guessing the majority class for a 
# 10 class problem when we have an example number of examples per class

clf = naive_bayes.BernoulliNB()
# you'll improve things by trying a new classifier here

cv = cross_validation.StratifiedKFold(y=y, n_folds=10, shuffle=True)

scores = cross_validation.cross_val_score(clf, X, y, cv=cv)

"Scores: {:0.3f} +/- {:0.3f} (2 s.d.)".format(scores.mean(), scores.std()*2)
# We expect approximately 88% accuracy

# We'll fix a random seed so the results are the same on each run,
# you wouldn't do this in practice but it does help when you're debugging
random_state = 47
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
_, X_test_greyscale, _, _ = train_test_split(X_greyscale, y, test_size=test_size, random_state=random_state)

print("Score on the training data:")
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

print("Score on the test split:")
print(clf.score(X_test, y_test))
# We expect both train and test scores to be pretty similar

predicted_y_test = clf.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, predicted_y_test)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  #, fig_kw = {"figsize":(6,4)})
#ax1.plot(x, y)
ax1.set_title('Precision')
ax1.set_ylabel("Percent")
ax2.set_title('Recall')

# plot bars, set 10 class labels
classes = np.arange(precision.shape[0])
_ = ax1.bar(classes, precision)
ax1.set_xticklabels([str(c) for c in classes])
_ = ax1.set_xticks(classes)
_ = ax2.bar(classes, recall)
ax2.set_xticklabels([str(c) for c in classes])
_ = ax2.set_xticks(classes)

def plot_confusion_matrix(df, cmap=plt.cm.Blues, title="Confusion matrix"):
    """Use Seaborn to plot the confusion matrix"""
    sns.heatmap(df, annot=True, vmin=0, vmax=100, cmap=cmap)
    plt.ylabel('True label')
    plt.yticks(rotation=90)
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.title(title)
    
cm = confusion_matrix(y_test, clf.predict(X_test))

col_names = [str(n) for n in range(10)]
cm_df = pd.DataFrame(data=cm, columns=col_names, index=col_names)
plot_confusion_matrix(cm_df, title="Confusion matrix for digits task")

# Get the indices for class 1 for our diagnosis
target_class = 1 # class we want to investigate
true_labels_1s_indices = y_test == target_class
predictions_with_1s_as_truth = clf.predict(X_test)[true_labels_1s_indices]
# Get the class predictions for every example of Class 1
predictions_probabilities_with_1s_as_truth = clf.predict_proba(X_test)[true_labels_1s_indices]

# Convert the 2D array of probabilities into a dictionary of column vectors    
cols_to_vectors_dict = {c:predictions_probabilities_with_1s_as_truth[:,c] for c in range(predictions_probabilities_with_1s_as_truth.shape[1])}
df_cols_to_vectors = pd.DataFrame(cols_to_vectors_dict)
    
def plot_classification_probabilities(df, title, cmap=plt.cm.Blues):
    """Plot 10 columns of class probabilities for all the examples in our dataframe"""
    sns.heatmap(df, annot=False, vmin=0, vmax=1, cmap=cmap, yticklabels=False)
    plt.ylabel('Examples')
    #plt.yticks(rotation=90)
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45)
    plt.title(title)
plot_classification_probabilities(df_cols_to_vectors, title="Probabilities of each class being classified\nfor labelled {} examples".format(target_class))

# use argsort to return an array of indices where the 0th item (index_to_investigate) represents the 
# index of the lowest item in predictions_probabilities_with_1s_as_truth (and the 1st item 
# is the next-lowest etc) and select the index of the least-confident prediction of class 1
index_to_investigate = 0  # you can ask for 0 to 9th least-confident (where 9==most confident) item
idx_least_confident_prediction_for_class_1 = predictions_probabilities_with_1s_as_truth[:,target_class].argsort()[index_to_investigate]
print("Index-{} least confident Class {} prediction is on row {} of predictions_with_1s_as_truth".format(index_to_investigate, target_class, idx_least_confident_prediction_for_class_1))

# for this one example we can get all 10 class probabilities
probabilities_per_class = predictions_probabilities_with_1s_as_truth[idx_least_confident_prediction_for_class_1]
print("Probabilities:\n", probabilities_per_class)
_ = plt.bar(np.arange(probabilities_per_class.shape[0]), probabilities_per_class)
plt.ylabel("Probabilities")
plt.title("Probabilities for each predicted class (we want class {})".format(target_class))

# the *most* confident class has the highest probability
idx_most_confident_class_for_this_example = probabilities_per_class.argsort()[-1]
print("We'd *incorrectly* predicted class {} for this example with probability {}".format(idx_most_confident_class_for_this_example, probabilities_per_class[idx_most_confident_class_for_this_example]))

idx_for_truth_for_this_example = np.arange(true_labels_1s_indices.shape[0])[true_labels_1s_indices][idx_least_confident_prediction_for_class_1]
show_example(idx_for_truth_for_this_example, X_test, y_test)

show_example(idx_for_truth_for_this_example, X_test_greyscale, y_test)



