# Import the libraries we will be using
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from dstools import data_tools

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 10, 8

# Load the data
data = pd.read_csv("data/mailing.csv")
# Let's take a look at the data
data.head()

for field in ['rfaa2', 'pepstrfl']:
    # Go through each possible value 
    for value in data[field].unique():
        # Create a new binary field
        data[field + "_" + value] = pd.Series(data[field] == value, dtype=int)

    # Drop the original field
    data = data.drop([field], axis=1)
    
    
data.head()

# Split our data
X = data.drop(['class'], axis=1)
Y = data['class']
X_mailing_train, X_mailing_test, Y_mailing_train, Y_mailing_test = train_test_split(X, Y, train_size=.75)

# Make and fit a model on the training data
model_mailing = LogisticRegression(C=1000000)
model_mailing.fit(X_mailing_train, Y_mailing_train)

# Get probabilities of being a (We saw this last class !!)
probabilities = model_mailing.predict_proba(X_mailing_test)[:, 1]

#print (model_mailing.predict_proba(X_mailing_test))

prediction = probabilities > 0.5

# Build and print a confusion matrix
confusion_matrix_large = pd.DataFrame(metrics.confusion_matrix(Y_mailing_test, prediction, labels=[1, 0]).T,
                                columns=['p', 'n'], index=['Y', 'N'])
print (confusion_matrix_large)

# Let's move the threshold down
prediction = probabilities > 0.05

# Build and print a confusion matrix
confusion_matrix_small = pd.DataFrame(metrics.confusion_matrix(Y_mailing_test, prediction, labels=[1, 0]).T,
                                columns=['p', 'n'], index=['Y', 'N'])
print (confusion_matrix_small)


X, Y = data_tools.handson_data()


#Split train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.8)

# Fit a logistic regression model
for c in [0.001, 0.01, 0.1, 1.0,10,100]:
    model = LogisticRegression(C=c)
    model.fit(X_train, Y_train)

    # Get the probability of Y_test records being = 1
    Y_test_probability_1 = model.predict_proba(X_test)[:, 1]

    # Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability_1)
    
    # Get the area under the curve (AUC)
    auc = np.mean(cross_val_score(model, X, Y, scoring="roc_auc"))

    # Plot the ROC curve
    plt.plot(fpr, tpr, label="AUC (C=" + str(c) + ") = " + str(round(auc, 2)))
    
plt.xlabel("False positive rate (fpr)")
plt.ylabel("True positive rate (tpr)")
plt.plot([0,1], [0,1], 'k--', label="Random")
plt.legend(loc=2)

# Fit a logistic regression model
model = LogisticRegression(C=1.0)
model.fit(X_train, Y_train)

# Get the predicted value and the probability of Y_test records being = 1
Y_test_predicted = model.predict(X_test)
Y_test_probability_1 = model.predict_proba(X_test)[:, 1]

# Sort these predictions, probabilities, and the true value in descending order of probability
order = np.argsort(Y_test_probability_1)[::-1]
Y_test_predicted_sorted = Y_test_predicted[order]
Y_test_probability_1_sorted = Y_test_probability_1[order]
Y_test_sorted = np.array(Y_test)[order]

# Go record-by-record and build the cumulative response curve
x_cumulative = []
y_cumulative = []
total_test_positives = np.sum(Y_test)
for i in range(1, len(Y_test_probability_1_sorted)+1):
    x_cumulative.append(i)
    y_cumulative.append(np.sum(Y_test_sorted[0:i]) / float(total_test_positives))

# Rescale
x_cumulative = np.array(x_cumulative)/float(np.max(x_cumulative)) * 100
y_cumulative = np.array(y_cumulative) * 100

# Plot
plt.plot(x_cumulative, y_cumulative, label="Classifier")
plt.plot([0,100], [0,100], 'k--', label="Random")
plt.xlabel("Percentage of test instances targeted (decreasing score)")
plt.ylabel("Percentage of positives targeted")
plt.title("Cumulative response curve")
plt.legend()

x_lift = x_cumulative
y_lift = y_cumulative/x_lift

plt.plot(x_lift, y_lift, label="Classifier")
plt.plot([0,100], [1,1], 'k--', label="Random")
plt.xlabel("Percentage of test instances (decreasing score)")
plt.ylabel("Lift (times)")
plt.title("Lift curve")
plt.legend()
plt.show()


X_train, X_test, Y_train, Y_test = X_mailing_train, X_mailing_test, Y_mailing_train, Y_mailing_test


# Keep track of all output with the name of the models
xs = {}
ys = {}
model_types = ["logistic_regression", "decision_tree"]

for model_type in model_types:
    # Instantiate the model
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(criterion="entropy", max_depth=15)
    elif model_type == "logistic_regression":
        model = LogisticRegression(C=1000000)
    model.fit(X_train, Y_train)

    Y_test_predicted = model.predict(X_test)
    Y_test_probability_1 = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability_1)
    # How many users are above the current threshold?
    n_targeted = []
    for t in thresholds:
        n_targeted.append(np.sum(Y_test_probability_1 >= t))

    # Turn these counts to percentages of users above the threshold
    n_targeted = np.array(n_targeted)/float(len(Y_test))

    # Store
    xs[model_type] = n_targeted * 100
    ys[model_type] = tpr * 100
    
    plt.plot(n_targeted * 100, tpr * 100, label=model_type)# * np.sum(Y_test)/float(len(Y_test)))

plt.plot([0,100], [0,100], 'k--', label="Random")
plt.xlabel("Percentage of test instances targeted (decreasing score)")
plt.ylabel("Percentage of positives targeted")
plt.title("Cumulative response curve")
plt.legend(loc=2)
plt.show()


for model_type in model_types:    
    # Previously computed: n_targeted * 100
    x_lift = xs[model_type]
    # Previously computed: tpr * 100
    y_lift = ys[model_type]/x_lift
    plt.plot(x_lift, y_lift, label=model_type)

plt.plot([0,100], [1,1], 'k--', label="Random")
plt.xlabel("Percentage of test instances targeted (decreasing score)")
plt.ylabel("Lift (times)")
plt.title("Lift curve")
plt.legend()
plt.ylim([.5,3])
plt.show()

cost_matrix = pd.DataFrame([[17, -1], [0, 0]], columns=['p', 'n'], index=['Y', 'N'])
print ("Cost matrix")
print (cost_matrix)

print ("Confusion matrix with threshold = 50% to predict labels")
print (confusion_matrix_large)
print ("\n")
print ("Confusion matrix with threshold = 5% to predict labels")
print (confusion_matrix_small)


print ("Expected profit per targeted individual with a cutoff of 50%% is $%.2f." % np.sum(np.sum(confusion_matrix_large/float(len(Y_test)) * cost_matrix)))
print ("Expected profit per targeted individual with a cutoff of 5%% is $%.2f." % np.sum(np.sum(confusion_matrix_small/float(len(Y_test)) * cost_matrix)))


X_train, X_test, Y_train, Y_test = X_mailing_train, X_mailing_test, Y_mailing_train, Y_mailing_test

# Make and fit a model on the training data
model = model_mailing

# Get the false positive rate, true positive rate, and all thresholds
fpr, tpr, thresholds = metrics.roc_curve(Y_test, model.predict_proba(X_test)[:,1])
Size_targeted_pop = float(len(Y_test))

# What is the baseline probability of being positive or negative in the data set?
p_p = np.sum(Y_test)/float(len(Y_test))
p_n = 1 - np.sum(Y_test)/float(len(Y_test))

Y_test_predicted = model.predict(X_test)
Y_test_probability_1 = model.predict_proba(X_test)[:, 1]

# How many users are above the current threshold?
n_targeted = []
for t in thresholds:
    n_targeted.append(np.sum(Y_test_probability_1 >= t))

# Turn these counts to percentages of users above the threshold
n_targeted = np.array(n_targeted)/float(len(Y_test))

# Expected profits:  
expected_profits = (cost_matrix['p']['Y']*(tpr*p_p)) + (cost_matrix['n']['Y']*(fpr*p_n))

# Plot the profit curve
plt.plot(n_targeted, Size_targeted_pop*expected_profits)
plt.xlabel("Percentage of users targeted")
plt.ylabel("Profit")
plt.title("Profits")

