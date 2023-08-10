import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
import sqlalchemy
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from IPython.core import display as ICD

from src.python import notebook_funcs as nb
get_ipython().magic('matplotlib inline')
sns.set()

conn = nb.connect('tremor')

# prepare a list of gesture names in the order they are processed
gestures = ['Lap Left', 'Lap Right', 'Nose Left', 'Nose Right', 'Shoulder Left', 'Shoulder Right']

# get all gesture data
all_gesture_dfs = nb.get_datasets(conn)

def make_pipe(reg=1000):
    # logistic regression hyperparameters
    lr_penalty = 'l1' # l1=lasso, l2=ridge
    lr_C = reg # C=1000 suppresses regularization, default is C=1.0
    lr_class_weight = 'balanced' # automatically adjust weights inversely proportional to class frequencies
    lr_max_iter = 100 # insensitive between 50 and 10000 - leave
    lr_solver = 'liblinear' # best solver for this problem
    lr_tol = 0.0001

    # build a logistic regression pipeline with scaling and regularization
    lr_scaler = StandardScaler(with_mean=True, with_std=True)
    lr = LogisticRegression(
            penalty=lr_penalty,
            C=lr_C, 
            class_weight=lr_class_weight, 
            max_iter=lr_max_iter, 
            solver=lr_solver,
            tol=lr_tol)

    lr_pipeline = make_pipeline(lr_scaler, lr)
    return [lr_pipeline, lr]

num_folds = 10 # optimal or close to it
cv_all_feats = [] # store cross-validation scores here for comparison later

for i, gesture in enumerate(all_gesture_dfs):
    X, y = nb.create_predictor_set(gesture)
    
    if i == 0:
        # this will be the same for all six sets, so print for set 0
        ne_rate = pd.Series.sum(y)/float(pd.Series.count(y))
        print "Null error rate for all data sets is {}\n".format(ne_rate)
    
    # make a new pipeline for each gesture
    (my_pipe, my_lr) = make_pipe()

    # evaluate model accuracy using k-fold cross validation
    cv_probabilities = cross_val_score(my_pipe, X, y, cv=num_folds)
    mean_acc = np.mean(cv_probabilities)
    cv_all_feats.append(mean_acc)
    
# compare model with restricted feature sets
improvement_list = pd.DataFrame(
    {'Gesture': gestures,
    'All features': cv_all_feats,
    'Improvement over Null Error rate': [x - ne_rate for x in cv_all_feats]
})

print "Cross-validation (k={})".format(num_folds)
improvement_list[['Gesture', 'All features', 'Improvement over Null Error rate']]
    

max_feats = 10

# the best feature set is the last one to increase accuracy
# over its predecessor of more than min_increment
min_increment = 0.001

for i, gesture in enumerate(all_gesture_dfs):

    print "Gesture {}".format(gestures[i])
    X, y = nb.create_predictor_set(gesture)
        
    # make a new pipeline for each gesture
    (my_pipe, my_lr) = make_pipe()
    
    (plot_x, plot_y) = nb.search_features(my_pipe, X, y, max_feats)
    print "Optimal feature set at k={}".format(nb.find_idx(plot_y, min_increment))  
            
    # graph to illustrate optimal number of features
    nb.sffs_plot(plot_x, plot_y)

optimal_feats = [['age','iqrx','tlagz'],
                 ['age','f0x','f0aa','tkeoy','meanaj','iqrx'],
                 ['age','q3z','p0fz'],
                 ['age','dfay','modex','zcrx'],
                 ['age','q1aa','zcrx','p0y'],
                 ['age','f0x','iqrz','sdy']]

cv_selected_feats = [] # store cross-validation scores for selected feature set

for i, gesture in enumerate(all_gesture_dfs):
    X, y = nb.create_predictor_set(gesture, optimal_feats[i])
    
    # re-evaluate model accuracy using k-fold cross validation
    # with the selected feature set
    cv_probabilities = cross_val_score(my_pipe, X, y, cv=num_folds)
    cv_selected_feats.append(np.mean(cv_probabilities))

# compare model with restricted feature sets
comparison_list = pd.DataFrame(
    {'Gesture': gestures,
     'All features': cv_all_feats,
     'Selected features': cv_selected_feats
    })

print "Cross-validation (k={})".format(num_folds)
comparison_list[['Gesture', 'All features', 'Selected features']]

for i, gesture in enumerate(all_gesture_dfs):
    
    train, test = train_test_split(gesture, test_size = 0.2)
    X_train = train[optimal_feats[i]]
    y_train = train['datagroups']
    X_test = test[optimal_feats[i]]
    y_test = test['datagroups']
    
    # make a new pipeline for each gesture
    (my_pipe, my_lr) = make_pipe()

    # generate data for the ROC curve
    my_pipe.fit(X_train, y_train)
    roc_probabilities = my_pipe.predict_proba(X_test)[:, 1]

    # build the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, roc_probabilities, pos_label=None, sample_weight=None, drop_intermediate=True)
    roc_auc = roc_auc_score(y_test, roc_probabilities)
    
    # plot the ROC curve
    nb.roc_on(fpr, tpr, roc_auc, gestures[i])
    
    # build and then display confusion matrix
    nb.build_cfm(roc_probabilities, y_test)
    

unrestricted_cv_scores = []

for i, gesture in enumerate(all_gesture_dfs):
    X, y = nb.create_predictor_set(gesture)
    highest_acc = [0,0]
    
    for reg_val in np.arange(0.1,0.25,0.01):
        (my_pipe, my_lr) = make_pipe(reg=reg_val)

        # evaluate model accuracy using k-fold cross validation
        cv_probabilities = cross_val_score(my_pipe, X, y, cv=num_folds)
        mean_acc = np.mean(cv_probabilities)
        if mean_acc > highest_acc[0]:
            highest_acc[0] = mean_acc
            highest_acc[1] = reg_val
    unrestricted_cv_scores.append(highest_acc)

# compare to models with restricted feature sets
best_reg_vals = [x[1] for x in unrestricted_cv_scores]
comparison_list = pd.DataFrame(
    {'Gesture': gestures,
     'Accuracy (selected features)': cv_selected_feats,
     'Best Accuracy (all features w. regularization)': [x[0] for x in unrestricted_cv_scores],
     'Best Regularization': best_reg_vals
    })

print "Cross-validation (k={})".format(num_folds)
comparison_list[[3,0,1,2]]

def pair_model_accuracy(df1,df2,f1='',f2='',reg1=1000,reg2=1000):
    '''
    Merges two gesture datasets and returns cross-validated classification accuracy

    Input:
      m1, m2 - (object) two gesture data frames to be merged
      f1, f2 - (list) two feature lists - use all features if empty
      reg1, reg2 - (decimal) regularization parameters - 1000 if null
    Output:
      10-fold cross-validated accuracy of pair model
    '''
    # create a set of predictors for this pair of gestures
    first_X, y = nb.create_predictor_set(df1, f1)
    second_X = nb.create_predictor_set(df2, f2)[0]

    # age will be in both X data frames - remove one copy
    first_X.drop('age', axis=1, inplace=True)

    # merge X data frames
    pair_X = pd.concat([first_X, second_X], axis=1)

    # make a pipeline: regularization is the mean of the reg values
    (my_pipe, my_lr) = make_pipe(reg=(reg1+reg2)/2)

    # cross-validate and return mean accuracy
    return np.mean(cross_val_score(my_pipe, pair_X, y, cv=num_folds))

ar = []
af = []

for m1,m2 in it.combinations([0,1,2,3,4,5], 2):
    # for all pairwise gesture combinations
    # calculate CV-10 accuracy using regularization only
    ar.append(pair_model_accuracy(all_gesture_dfs[m1],all_gesture_dfs[m2],reg1=best_reg_vals[m1],reg2=best_reg_vals[m2]))
    # calculate CV-10 accuracy using feature selection only
    af.append(pair_model_accuracy(all_gesture_dfs[m1],all_gesture_dfs[m2],f1=optimal_feats[m1],f2=optimal_feats[m1]))

print "CV-10 accuracy using regularization only: {:.6f} (sd={:.6f})".format(np.mean(ar),np.std(ar))
print "CV-10 accuracy using feature selection only: {:.6f} (sd={:.6f})".format(np.mean(af),np.std(af))





