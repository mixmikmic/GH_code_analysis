get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

projects = pd.DataFrame.from_csv('opendata_projects.csv', index_col=None)

projects.head()

projects.funding_status.unique()

## Some Data wrangling
projects = projects[(projects.total_price_including_optional_support > 0) & (projects.funding_status == 'completed') | (projects.funding_status == 'expired')]
projects['date_posted'] = pd.to_datetime(projects['date_posted'])
projects['year'] = projects['date_posted'].dt.year
projects['month'] = projects['date_posted'].dt.month
projects['date_posted_1'] = projects['date_posted'].map(lambda x: 100*x.year + x.month)

# Label Encoding categories
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
proj = projects.apply(le.fit_transform)

proj.head()

len(proj)

# Categorical features
# Using OnHotEncode for categorical features
enc = preprocessing.OneHotEncoder()
a = enc.fit_transform(proj[['school_metro','primary_focus_subject','school_state','poverty_level','grade_level','resource_type','year','month']]).toarray()

# Boolean features
b = proj.as_matrix([['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise',
                     'eligible_double_your_impact_match','eligible_almost_home_match','teacher_teach_for_america',
                    'teacher_ny_teaching_fellow']])

# Numerical feature
z = projects.as_matrix([['total_price_including_optional_support']])

## Concatenate all features to use as input to model
c = np.concatenate((a,b,z),axis=1)

c

pd.DataFrame(c).head()

#proj_input = proj[['school_charter','school_magnet','school_year_round','school_nlns','school_kipp','school_charter_ready_promise']]
proj_input = c
proj_output = projects['funding_status']
#proj_input = proj_input.replace('f',0).replace('t',1)

from sklearn import tree
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(proj_input,proj_output)

from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier()
clf2 = clf2.fit(proj_input,proj_output)

projects['predicted_by_clf1'] = clf1.predict(proj_input)
projects['predicted_by_clf2'] = clf2.predict(proj_input)

projects[projects.funding_status == 'expired'][['funding_status','predicted_by_clf1','predicted_by_clf2','total_price_including_optional_support']].head()

len(projects[projects.funding_status == 'expired'])

a = pd.DataFrame(clf1.predict(proj_input))

clf1.predict(proj_input)

tree.export_graphviz(clf1,out_file='clf1_4.dot')     



# special IPython command to prepare the notebook for matplotlib
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Generic classification and optimization functions from last lab
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# clf - original classifier
# parameters - grid to search over
# X - usually your training X matrix
# y - usually your training y 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print "BEST", gs.best_params_, gs.best_score_, gs.grid_scores_
    best = gs.best_estimator_
    return best

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Important parameters
# indf - Input dataframe
# featurenames - vector of names of predictors
# targetname - name of column you want to predict (e.g. 0 or 1, 'M' or 'F', 
#              'yes' or 'no')
# target1val - particular value you want to have as a 1 in the target
# mask - boolean vector indicating test set (~mask is training set)
# reuse_split - dictionary that contains traning and testing dataframes 
#              (we'll use this to test different classifiers on the same 
#              test-train splits)
# score_func - we've used the accuracy as a way of scoring algorithms but 
#              this can be more general later on
# n_folds - Number of folds for cross validation ()
# n_jobs - used for parallelization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask=None, reuse_split=None, score_func=None, n_folds=5, n_jobs=1):
    #subdf=indf[featurenames]
    X=featurenames#subdf.values
    y=(indf[targetname].values==target1val)*1
    if mask !=None:
        print "using mask"
        Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
    if reuse_split !=None:
        print "using reuse split"
        Xtrain, Xtest, ytrain, ytest = reuse_split['Xtrain'], reuse_split['Xtest'], reuse_split['ytrain'], reuse_split['ytest']
    if parameters:
        clf = cv_optimize(clf, parameters, Xtrain, ytrain, n_jobs=n_jobs, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (training_accuracy)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print confusion_matrix(ytest, clf.predict(Xtest))
    print "########################################################"
    return clf, Xtrain, ytrain, Xtest, ytest

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Plot tree containing only two covariates
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

from matplotlib.colors import ListedColormap
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def plot_2tree(ax, Xtr, Xte, ytr, yte, clf, plot_train = True, plot_test = True, lab = ['Feature 1', 'Feature 2'], mesh=True, colorscale=cmap_light, cdiscrete=cmap_bold, alpha=0.3, psize=10, zfunc=False):
    # Create a meshgrid as our test data
    plt.figure(figsize=(15,10))
    plot_step= 0.05
    xmin, xmax= Xtr[:,0].min(), Xtr[:,0].max()
    ymin, ymax= Xtr[:,1].min(), Xtr[:,1].max()
    xx, yy = np.meshgrid(np.arange(xmin, xmax, plot_step), np.arange(ymin, ymax, plot_step) )

    # Re-cast every coordinate in the meshgrid as a 2D point
    Xplot= np.c_[xx.ravel(), yy.ravel()]


    # Predict the class
    Z = clfTree1.predict( Xplot )

    # Re-shape the results
    Z= Z.reshape( xx.shape )
    cs = plt.contourf(xx, yy, Z, cmap= cmap_light, alpha=0.3)
  
    # Overlay training samples
    if (plot_train == True):
        plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr-1, cmap=cmap_bold, alpha=alpha,edgecolor="k") 
    # and testing points
    if (plot_test == True):
        plt.scatter(Xte[:, 0], Xte[:, 1], c=yte-1, cmap=cmap_bold, alpha=alpha, marker="s")

    plt.xlabel(lab[0])
    plt.ylabel(lab[1])
    plt.title("Boundary for decision tree classifier",fontsize=7.5)

len(proj)

# Create test/train mask
itrain, itest = train_test_split(xrange(projects.shape[0]), train_size=0.60)
mask=np.ones(projects.shape[0], dtype='int')
mask[itrain]=1
mask[itest]=0
mask = (mask==1)

print "% Project Success in training data:", np.mean(proj.funding_status[mask])
print "% Project Success in test data:", np.mean(proj.funding_status[~mask])

from sklearn import tree
clf = tree.DecisionTreeClassifier()

parameters = {"max_depth": [None], 'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
#parameters = {}
clf,Xtrain_dt,ytrain_dt,Xtest_dt,ytest_dt = do_classify(clf,parameters,projects,c,'funding_status','completed',mask=mask,
                                           n_jobs = 4,score_func = 'f1')

# Depth of a decision tree
clf.tree_.max_depth

clf.predict(Xtest_dt)

# Plot of a ROC curve for a specific class
y_score_dt = pd.DataFrame(clf.predict_proba(Xtest_dt))[1]
import sklearn.metrics as metric
fpr,tpr,thresholds = metric.roc_curve(ytest_dt,y_score_dt)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()

print 'Area under ROC: ',metric.roc_auc_score(ytest_dt,y_score_dt)
print 'Precision, Recall, Fscore:'
metric.precision_recall_fscore_support(ytest_dt,clf.predict(Xtest_dt),average='binary')

from sklearn.ensemble import RandomForestClassifier
clfForest = RandomForestClassifier()

parameters = {"n_estimators": [19], "min_samples_split":[1,2,4,6]}
#parameters = {"n_estimators": range(1, 20)}
clfForest, Xtrain, ytrain, Xtest, ytest = do_classify(clfForest, parameters, 
                                                       projects, c,'funding_status', 'completed', mask=mask, 
                                                       n_jobs = 4, score_func='f1')

# Depths of decision tress inside random forest classifier
[estimator.tree_.max_depth for estimator in clfForest.estimators_]

print len(proj)
print len(proj[proj.funding_status == 1])
print len(proj[proj.funding_status == 0])

# Plot of a ROC curve for a specific class
y_score = pd.DataFrame(clfForest.predict_proba(Xtest))[1]
import sklearn.metrics as metric
fpr,tpr,thresholds = metric.roc_curve(ytest,y_score)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()

print 'Area under ROC: ',metric.roc_auc_score(ytest,y_score)
print 'Precision, Recall, Fscore:'
metric.precision_recall_fscore_support(ytest,clfForest.predict(Xtest),average='binary')

print 'Area under ROC: ',metric.roc_auc_score(ytest,y_score,average ='macro')

# Logistic regression
from sklearn.linear_model import LogisticRegression
clfLog = LogisticRegression()

#parameters = {"multi_class": ['ovr'],"solver":['newton-cg','lbfgs','liblinear','sag']}
parameters = {"C":[0.01,0.1,1,10,100,1000]}
clfLog, Xtrain_log, ytrain_log, Xtest_log, ytest_log = do_classify(clfLog, parameters, 
                                                       projects, c,'funding_status', 'completed', mask=mask, 
                                                       n_jobs = 4, score_func='f1')

clfLog.predict_proba(Xtest_log)

clfLog.predict(Xtest_log)

# Plot of a ROC curve for a specific class
y_score = pd.DataFrame(clfLog.predict_proba(Xtest_log))[1]
import sklearn.metrics as metric
fpr,tpr,thresholds = metric.roc_curve(ytest_log,y_score)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Curve')
plt.legend(loc="lower right")
plt.show()

print 'Area under ROC: ',metric.roc_auc_score(ytest_log,y_score)
print 'Precision, Recall, Fscore:'
metric.precision_recall_fscore_support(ytest_log,clfLog.predict(Xtest_log), average='binary')

best_ratio = 0
for i,threshold in enumerate(thresholds):
    ratio = tpr[i]/fpr[i]
    if (ratio > best_ratio) & (ratio != np.inf):
        best_ratio = ratio
        best_threshold = threshold
        best_index = i

print best_ratio,best_threshold, best_index

tpr[3]

thresholds[3]

# TODO: Need to check, not sure
importance_list = clfForest.feature_importances_
name_list = proj[['total_price_including_optional_support','school_metro','primary_focus_subject','school_state','poverty_level','grade_level','resource_type',
                 'school_charter_ready_promise','eligible_double_your_impact_match','eligible_almost_home_match','teacher_teach_for_america','teacher_ny_teaching_fellow']].columns
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))
plt.barh(range(len(name_list)),importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)
plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature')
plt.show()

from sklearn.ensemble import AdaBoostClassifier
clfAda = AdaBoostClassifier()

parameters = {"n_estimators": range(10, 60)}
clfAda, Xtrain, ytrain, Xtest, ytest = do_classify(clfAda, parameters, 
                                                       proj, c,'funding_status', 1, mask=mask, 
                                                       n_jobs = 4, score_func='f1')

