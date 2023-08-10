# We will need following libraries
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # So that display() can be used for Dataframe
import visuals as vs
get_ipython().magic('matplotlib inline')

# Loading the ILPD dataset
data = pd.read_csv("ILPD.csv")

print "No of samples: {}. No of features in each sample: {} .".format(*data.shape)
# Display the first  5 records
display(data.head(n=5))

#some statistics about the data
display(data.describe())

# We can see that the column 'Alkphos' has 4 missing values
alkphos=data['Alkphos']
print "length before removing NaN values:%d"%len(data)
data = data[pd.notnull(data['Alkphos'])]
print "length after removing NaN values:%d"%len(data)

# Split the data into features and target label(disease)
disease_initial = data['Disease']
features_initial = data.drop('Disease', axis = 1)

# Visualize skewed continuous features of original data
import matplotlib.pyplot as plt
pd.options.display.mpl_style = 'default'
features_initial.hist(figsize=(14,8))

# Skewed features are Albumin, Direct Bilirubin, A/G ratio, Tota Bilirubin, Total Protein 
#Log-transform the skewed features
skewed = ['Albumin', 'Direct Bilirubin', 'Total Bilirubin', 'A/G ratio', 'Total Proteins']
features_initial[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
features_initial.hist(figsize=(14,8))

# Import sklearn.preprocessing.StandardScaler- producing  values in range of 10^-16
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()

normalized = ['Age', 'SGPT', 'SGOT', 'Alkphos', 'Albumin', 'Direct Bilirubin', 'Total Bilirubin', 'A/G ratio', 'Total Proteins']
skewed = ['Albumin', 'Direct Bilirubin', 'Total Bilirubin', 'A/G ratio', 'Total Proteins']
features_initial[normalized] = scaler.fit_transform(data[normalized])

# Show an example of a record with scaling applied
display(features_initial.describe())

# TODO: One-hot encode the data using pandas.get_dummies()
features = pd.get_dummies(features_initial)

encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))
print encoded
display(features.head(n = 1))

disease=pd.get_dummies(disease_initial)
encoded = list(disease.columns)
print "{} disease columns after one-hot encoding.".format(len(encoded))
#print disease[1]
display(disease.head(n = 1))

# Import train_test_split
from sklearn.cross_validation import train_test_split, ShuffleSplit

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, disease[1], test_size = 0.2, random_state = 7)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#naive accuracy, assuming that we predict everyone has disease
positive_disease=len(data[data['Disease'] == 1])
print positive_disease
accuracy = positive_disease*1.0/len(data)
precision=accuracy
recall=1
beta=2# assuming beta to be 2, giving greater weightage to recall
fscore = (1+beta*beta)*precision*recall/((beta*beta*precision)+recall)
# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

# Here I use some portions of the codebase of the MLND supervised learning project 'finding_donors'
from sklearn.metrics import fbeta_score, accuracy_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    results = {}

    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    results['train_time'] = end - start
        
    start = time() 
    predictions_test = learner.predict(X_test)# predictions on test set
    predictions_train = learner.predict(X_train[:200])# predictions on first 200 elements of training set
    end = time()
    
    results['pred_time'] = end - start
            
    results['acc_train'] = accuracy_score(y_train[:200],predictions_train)
        
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    results['f_train'] = fbeta_score(y_train[:200],predictions_train,beta=2)
        
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=2)
       
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    return results

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# TODO: Initialize the three models

clf_base = LogisticRegression(random_state=7)
clf_A = RandomForestClassifier(random_state=7)
clf_B = SVC(random_state=7)
clf_C = KNeighborsClassifier()

# TODO: Calculate the number of samples for 20%, 50%, and 100% of the training data
samples_20 = int(len(X_train) * 0.2)
samples_50 = int(len(X_train) * 0.5)
samples_100 = int(len(X_train) )
result_1={}
result_1=train_predict(clf_base, samples, X_train, y_train, X_test, y_test)
print 'Performance metrics for benchmark model (Logistic regression):'
print 'Accuracy score on training subset:%.2f'%result_1['acc_train']
print 'Accuracy score on test subset:%.2f'%result_1['acc_test']
print 'F-score on training subset:%.2f'%result_1['f_train']
print 'F-score on test subset:%.2f'%result_1['f_test']
# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_20, samples_50, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Visuals borrowed from 'finding_donors' project
vs.evaluate(results, accuracy, fscore)

#plotting ROC curve

for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    pred=clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    print "For classifier %s, ROC score is %f"%(clf_name,roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
#fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)


from sklearn.metrics import roc_curve,auc
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
# TODO: Initialize the classifier
i=0
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    print 'For classifier %s:'%(clf_name)
    if i==0: #RandomForest
        parameters = {'max_features':['auto',None],# 'auto' option uses square root of number of features
                     'oob_score':[False,True],# setting it to 'True' saves the generalization error
                     'max_depth':[3,10,15],# depth of tree
                     'n_estimators':[3,10,15]}# number of trees
    elif i==1: #SVM
        parameters={'kernel':['poly','rbf','linear'],# different ways to separate data pts by a hyperplane
                    'C':[0.001,1,1000]} # weight of penalty assigned to error
    elif i==2: #kNearestClassifier
        parameters={'n_neighbors':[5,10,15],# number of neighbors
                    'weights':['uniform','distance']}# distance means weights are inversely proportional to distance

    scorer = make_scorer(fbeta_score, beta=2)
    grid_obj = GridSearchCV(clf,parameters,scoring=scorer)
    grid_fit = grid_obj.fit(X_train,y_train)
    best_clf = grid_fit.best_estimator_

    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)
    # Report the before-and-afterscores
    print "Unoptimized model\n------"
    print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
    print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 2))
    print "\nOptimized Model\n------"
    print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
    print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 2))
    print "Best parameters:"
    print grid_fit.best_params_
    fpr, tpr, _ = roc_curve(y_test, best_predictions)
    roc_auc = auc(fpr, tpr)
    print "For classifier %s, ROC score is %f"%(clf_name,roc_auc)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    print "\n---------------------------------X----------------------------\n"
    i+=1

clf_new=clf_A.fit(X_train,y_train)#clf_A is RandomForestClassifier
im_features = clf_new.feature_importances_
vs.feature_plot(im_features, X_train, y_train)

from sklearn.base import clone

X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

clf = (clone(clf_A)).fit(X_train_reduced, y_train)#seeing how using only 5 best features affects RandomForestClassifier
predictions=clf_A.predict(X_test)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from Random Forest using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))



