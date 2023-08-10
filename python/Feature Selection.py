get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

directory = 'UCI HAR Dataset/train/X_train.txt'
sub_dir = 'UCI HAR Dataset/train/subject_train.txt'
y_dir = 'UCI HAR Dataset/train/y_train.txt'
train = np.loadtxt(directory)
train_subject = np.loadtxt(sub_dir)
train_y = np.loadtxt(y_dir)

directory = 'UCI HAR Dataset/test/X_test.txt'
sub_dir = 'UCI HAR Dataset/test/subject_test.txt'
y_dir = 'UCI HAR Dataset/test/y_test.txt'
test = np.loadtxt(directory)
test_subject = np.loadtxt(sub_dir)
test_y = np.loadtxt(y_dir)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
#                           n_redundant=2, n_repeated=0, n_classes=8,
#                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=KFold(2),
              scoring='accuracy')
rfecv.fit(train, train_y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.title('Number of Features VS CV Accuracy')
plt.xlabel("Number of features selected")
plt.ylabel("CV Accuracy")
plt.plot(range(0, 561, 1), rfecv.grid_scores_)
plt.show()

#retrieve feature index in the original training dataset
features = []
for i in range(len(rfecv.support_)):
    if rfecv.support_[i]:
        features.append(i)

len(features)

reduced_train = train[:,rfecv.support_]

reduced_train.shape

#use cross validation and repeated random forest process to pick final features
final_scores = np.zeros(reduced_train.shape[1])
# for each cross validation set k fold, where k = 10
# repeat random forest process for n times, store importances scores into a list, n = 10
K = 1; N = 10;
for k in range(K):
    #subjects = list(set(train_subject))
    #sample = np.array([subjects[k], subjects[k+10]])
    #indx = np.array([x in sample for x in train_subject])
    #tn = reduced_train[indx != True]
    #tn_y = train_y[indx != True]
    #cv = reduced_train[indx]
    #cv_y = train_y[indx]
    for n in range(N):
        state = int(np.random.rand(1)*1000)
        #random forest
        forest = RandomForestClassifier(max_depth = None,
                                        min_samples_split=5,
                                        n_estimators = 1000,
                                        random_state = state,
                                        bootstrap = True, oob_score = True)
        my_forest = forest.fit(reduced_train, train_y)
        print "oob score: ", str(my_forest.oob_score_)
        final_scores += my_forest.feature_importances_
final_scores = final_scores/(K*N)

#feature selection, greed method
#sort features by importance scores,
#pick the most important feature,
#try every pair of this feature with each of other features, pick the best pair according to accuracy
#with two-feature group picked, repeat the process to find the third one.
#stop until five features are selected.
cv_fold = 10
subjects = list(set(train_subject))
K = 10
importances = list(final_scores)
s = sorted(importances, reverse = True)
best_feature =features[importances.index(s[0])]
best_group = [best_feature]


for k in range(K):
    #best_accuracy = 0.
    group = best_group
    features_dict = dict()
    for i in range(cv_fold):
        sample = subjects[i*2:i*2+2]
        #sample = np.random.choice(subjects, 2)
        indx = np.array([x in sample for x in train_subject])
        tn = train[indx != True]
        tn_y = train_y[indx != True]
        cv = train[indx]
        cv_y = train_y[indx]
        next_best_feature = None
        best_sub_accuracy = 0.
        for feature in features:
            if feature != best_feature:
                reduced_tn = tn[:,group+[feature]]
                forest = RandomForestClassifier(max_depth = None, min_samples_split=10,
                                                n_estimators = 50, random_state = 1)
                my_forest = forest.fit(reduced_tn,tn_y)
                reduced_cv = cv[:,group+[feature]]
                score = my_forest.score(reduced_cv,cv_y)
                if score > best_sub_accuracy:
                    best_sub_accuracy = score
                    next_best_feature = feature
        features_dict[next_best_feature] = features_dict.get(next_best_feature, 0) + 1
    bf = [x for x in features_dict.keys() if features_dict[x] == max(features_dict.values())][0]
    best_group += [bf]
    sample = np.random.choice(subjects, 3)
    indx = np.array([x in sample for x in train_subject])
    tn = train[indx != True]
    tn_y = train_y[indx != True]
    cv = train[indx]
    cv_y = train_y[indx]
    tn = tn[:,best_group]
    cv = cv[:,best_group]
    forest = RandomForestClassifier(max_depth = None, min_samples_split=10,
                                                n_estimators = 100, random_state = 1)
    my_forest = forest.fit(tn,tn_y)
    score = my_forest.score(cv,cv_y)
    print "score", score
    """    if best_sub_accuracy > best_accuracy:
        print "best_sub_accuracy "+str(best_sub_accuracy) + " is greater than best_accuracy " + str(best_accuracy)
    else:
        print "Warning, best_sub_accuracy is NOT greater than previous best_accuracy"""
    #best_accuracy = best_sub_accuracy
    feature_number = len(best_group)
    print "best group, number: ", str(feature_number)
    print "best features: ", best_group
    #print "best accuracy: ", best_accuracy
    
            

import seaborn as sns


def draw_scatter_plot(feature_indx, data, response):
    activity = []
    for y in response:
        if y == 1.0:
            activity.append('WALKING')
        elif y == 2.0:
            activity.append('WALKING_UPSTAIRS')
        elif y == 3.0:
            activity.append('WALKING_DOWNSTAIRS')
        elif y == 4.0:
            activity.append('SITTING')
        elif y == 5.0:
            activity.append('STANDING')
        elif y == 6.0:
            activity.append('LAYING')
    indx = []
    feature_name = []
    for line in open('UCI HAR Dataset\\features.txt','r'):
        line = line.split()
        l, f = line
        indx.append(l)
        feature_name.append(f)
    chosen_feature = []
    for ind in feature_indx:
        chosen_feature.append(feature_name[ind])
    print chosen_feature
    reduced_data = pd.DataFrame(data[:, feature_indx])
    reduced_data.columns = chosen_feature
    reduced_data['activity'] = activity
    sns.set()
    sns.pairplot(reduced_data, hue='activity', size = 6)

draw_scatter_plot([52, 504, 53, 69, 179], train, train_y)

forest = RandomForestClassifier(max_depth = None, min_samples_split=5, n_estimators = 100, random_state = 1)
reduced_train = train[:,[52, 504, 53, 69, 179]]
my_reduced_forest = forest.fit(reduced_train, train_y)
reduced_test = test[:,[52, 504, 53, 69, 179]]
my_reduced_forest.score(reduced_test, test_y)

print(__doc__)

import itertools
from sklearn.metrics import confusion_matrix

# import some data to play with
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
class_names = ["walking","walking upstairs", "walking downstairs", "sitting", "standing", "laying"]
y_pred = my_reduced_forest.predict(reduced_test)
# Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
#classifier = svm.SVC(kernel='linear', C=0.01)
#y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure(figsize=(8,8))
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      #title='Normalized confusion matrix')

plt.show()

