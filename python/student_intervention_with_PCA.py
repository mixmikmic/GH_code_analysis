get_ipython().magic(u'matplotlib inline')
# Import libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from time import time
from __future__ import division
from sklearn import cross_validation

from sklearn.metrics import f1_score

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
n_students = student_data.shape[0]
n_features = student_data.shape[1]-1
n_passed = student_data.loc[student_data['passed']=='yes'].shape[0]
n_failed = n_students-n_passed
grad_rate = 100*n_passed/n_students
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows

# Preprocess feature columns
def preprocess_features(X):
    
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty
    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

#plt.scatter(X_all[:,0],X_all[:,1],color='r')

# First, decide how many training vs test samples you want
#num_all = student_data.shape[0]  # same as len(student_data)
#num_train = 300  # about 75% of the data
#num_test = num_all - num_train
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.25, random_state=0)


from sklearn.decomposition import RandomizedPCA

# Reduce features
pca = RandomizedPCA(whiten=True).fit(X_train)
print "Visualizing all features"
pc_df=pd.DataFrame({"pca":pca.explained_variance_ratio_})
plt.plot(pc_df)
plt.ylabel('pca (exp vari. ratios)')
plt.xlabel('features')
plt.show()

print "Since variance becomes negligible around 30, hence will use n_components=30 for pca"

# Reduce features
pca = RandomizedPCA(n_components=25,whiten=True).fit(X_train)

t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "done in %0.5fs" % (time() - t0)
# TODO: Then, select features (X) and corresponding labels (y) for the training and test sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset

print "Training set: {} samples".format(X_train_pca.shape)
print "Test set: {} samples".format(X_test_pca.shape)
# Note: If you need a validation set, extract it from within training data

test_f1score_matrix=pd.DataFrame(columns=['SVC','AB','RF'],index=['110','130','150','170','190','210','230','250','270','290'])

# Train a model
import time

def train_classifier(clf, X_train, y_train):
    print "Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print "Done!\nTraining time (secs): {:.5f}".format(end - start)

# TODO: Choose a model, import it and instantiate an object
from sklearn.svm import SVC
clf = SVC(C=10,gamma=0.001)


# Fit model to training data
train_classifier(clf, X_train_pca, y_train)  # note: using entire training set here
#print clf  # you can inspect the learned model by printing it

print clf

# Predict on training set and compute F1 score
from sklearn.metrics import f1_score

def predict_labels(clf, features, target):
    print "Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    print "Done!\nPrediction time (secs): {:.5f}".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

train_f1_score = predict_labels(clf, X_train_pca, y_train)
print "F1 score for training set: {}".format(train_f1_score)

# Predict on test data
print "F1 score for test set: {}".format(predict_labels(clf, X_test_pca, y_test))

# Train and predict using different training set sizes
def train_predict(clf, X_train, y_train, X_test, y_test):
    print "------------------------------------------"
    print "Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print "F1 score for training set: {}".format(predict_labels(clf, X_train, y_train))
    test_score=predict_labels(clf, X_test, y_test)
    print "F1 score for test set: {}".format(test_score)
    return test_score

for i in range(1,11):
    clf = SVC(C=10,gamma=0.001)
    test_f1score_matrix['SVC'][i-1]=train_predict(clf, X_train_pca[0:90+(20*i)], y_train[0:90+(20*i)],X_test_pca, y_test)



# TODO: Train and predict using two other models
## Using AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#dt_clf=DecisionTreeClassifier(presort=True, min_samples_leaf=2,max_depth=1,max_features=15)
#clf = AdaBoostClassifier(dt_clf,n_estimators=10,learning_rate=0.1)
# Fit model to training data
#train_classifier(clf, X_train_pca, y_train)  # note: using entire training set here
#print clf

#train_f1_score = predict_labels(clf, X_train_pca, y_train)
#print "F1 score for training set: {}".format(train_f1_score)

# Predict on test data
#print "F1 score for test set: {}".format(predict_labels(clf, X_test_pca, y_test))

learning_rate=1.0
for i in range(1,11):
    dt_clf=DecisionTreeClassifier(presort=True, min_samples_leaf=1,max_depth=1,max_features=25)
    if i > 6:
        learning_rate=0.9
    clf = AdaBoostClassifier(dt_clf,algorithm="SAMME",n_estimators=80,learning_rate=learning_rate)
    test_f1score_matrix['AB'][i-1]=train_predict(clf, X_train_pca[0:90+(20*i)], y_train[0:90+(20*i)],X_test_pca, y_test)
#test_f1score_matrix['AB'][1]=train_predict(clf, X_train_pca[0:5], y_train[0:5],X_test_pca, y_test)
#test_f1score_matrix['AB'][2]=train_predict(clf, X_train_pca[0:100], y_train[0:100],X_test_pca, y_test)
#test_f1score_matrix['AB'][3]=train_predict(clf, X_train_pca[0:200], y_train[0:200],X_test_pca, y_test)
#test_f1score_matrix['AB'][4]=train_predict(clf, X_train_pca[0:290], y_train[0:290],X_test_pca, y_test)

# TODO: Train and predict using two other models
## Using RandomForest
from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators=10, max_depth=2,min_samples_leaf=2,max_features=15)
# Fit model to training data
#train_classifier(clf, X_train_pca, y_train)  # note: using entire training set here
#print clf

#train_f1_score = predict_labels(clf, X_train_pca, y_train)
#print "F1 score for training set: {}".format(train_f1_score)

# Predict on test data
#print "F1 score for test set: {}".format(predict_labels(clf, X_test_pca, y_test))


for i in range(1,11):
    clf = RandomForestClassifier(n_estimators=100, max_depth=3,min_samples_leaf=2,max_features=25)
    test_f1score_matrix['RF'][i-1]=train_predict(clf, X_train_pca[0:90+(20*i)], y_train[0:90+(20*i)],X_test_pca, y_test)


print test_f1score_matrix
test_f1score_matrix['count']=[110,130,150,170,190,210,230,250,270,290]
print "svc ", test_f1score_matrix[0:1].values
print "values: ", test_f1score_matrix.index

print test_f1score_matrix
plt.plot(test_f1score_matrix['count'].values,test_f1score_matrix['SVC'].values,label='SVC')
plt.plot(test_f1score_matrix['count'].values,test_f1score_matrix['AB'].values,label='AB')
plt.plot(test_f1score_matrix['count'].values,test_f1score_matrix['RF'].values,label='RF')
plt.legend(loc='upper left', shadow=True)
plt.show()

# TODO: Fine-tune your model and report the best F1 score

