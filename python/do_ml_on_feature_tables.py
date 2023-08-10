get_ipython().magic('matplotlib inline')

from __future__ import division
from __future__ import print_function
import csv
import datetime as dt
import os
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import clone
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

def csv_to_dict_cesar(csv_filename):
    # Let's say, We are intersted in only count features
    count_features = ['_char_count', '_hashtag_count', '_word_count', '_url_count']
    with open(csv_filename) as f:
        features = [({k: int(v) for k, v in row.items() if k in count_features}, row['_popular'])
                    for row in csv.DictReader(f, skipinitialspace=True)]
        X = [f[0] for f in features]
        Y = [f[1] for f in features]
    return (X, Y)

def csv_to_dict(csv_filename):
    """Open feature table with csv library.
    
    Task: Run with '_rt_count'. See the good results!
    """
    non_numeric_features = ['', '_text', '_urls', '_mentions', '_hashtags', 
                            '_tweet_datetime', '_popular', '_rt_count']
    with open(csv_filename, 'rU') as f:
        rows = csv.DictReader(f, skipinitialspace=True, delimiter='|')
        labels = [row['_popular'] for row in rows]

    features = []
    with open(csv_filename, 'rU') as f:
        rows = csv.DictReader(f, skipinitialspace=True, delimiter='|')
        for row in rows:
            #print(row)
            row_dict = {}
            for k, v in row.items():
                if k not in non_numeric_features:
                    try:
                        row_dict[k] = int(v)
                    # these tries catch a few junk entries
                    except TypeError:
                        row_dict[k] = 0
                    except ValueError:
                        row_dict[k] = 0

            #row_dict = {k: int(v) for k, v in row.items() if k not in non_numeric_features}
            features.append(row_dict)                

    return features, labels

def csv_to_df(csv_file):
    """Open csv with Pandas DataFrame, then convert to dict 
    and return.
    
    TODO: Fix this.
    """
    
    dataframe = pandas.read_csv(csv_file, 
                                encoding='utf-8', 
                                engine='python', 
                                sep='|',
                                delimiter='|',
                                index_col=0)
    return dataframe

def load_data(csv_filename):
    """Open csv file and load into Scikit vectorizer.
    """
    
    # Open .csv and load into df
    #features = csv_to_dict_cesar(csv_filename)
    #vec = DictVectorizer()
    #data = features[0]  # list of dict: [{'_word_count': 5, '_hashtag_count': 0, '_char_count': 50, '_url_count': 0}
    #target = features[1]  # list of str: ['TRUE', 'TRUE', 'FALSE', ...]
    
    print('Loading CSV into dict ...')
    t0 = dt.datetime.utcnow()
    data, target = csv_to_dict(csv_filename)
    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()
    
    print('Loading dict into vectorizer')
    t0 = dt.datetime.utcnow()
    vec = DictVectorizer()
    X = vec.fit_transform(data).toarray()  # change to numpy array
    Y = np.array(target)  # change to numpy array
    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()

    
    '''
    -In case we need to know the features
    '''
    feature_names = vec.get_feature_names()

    '''
    -Dividing the data into train and test
    -random_state is pseudo-random number generator state used for
     random sampling
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_data("feature_tables/basics.csv")

def scale_data(X_train, X_test, Y_train, Y_test):
    """Take Vectors, 
    """
    # write models dir if not present
    models_dir = 'models'
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    '''
    -PREPOCESSING 
    -Here, scaled data has zero mean and unit varience
    -We save the scaler to later use with testing/prediction data
    '''
    print('Scaling data ...')
    t0 = dt.datetime.utcnow()
    scaler = preprocessing.StandardScaler().fit(X_train)
    joblib.dump(scaler, 'models/scaler.pickle')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()

    return X_train_scaled, X_test_scaled, Y_train, Y_test

X_train_scaled, X_test_scaled, Y_train, Y_test = scale_data(X_train, X_test, Y_train, Y_test)

def run_tree(X_train_scaled, X_test_scaled, Y_train, Y_test):
    """Run decision tree with scikit.
    
    Experiment with: 'max_depth'
    """
    '''
    -This is where we define the models with pre-defined parameters
    -We can learn these parameters given our data
    '''
    print('Defining and fitting models ...')
    t0 = dt.datetime.utcnow()   
    dec_tree = DecisionTreeClassifier()

    dec_tree.fit(X_train_scaled, Y_train)

    joblib.dump(dec_tree, 'models/tree.pickle')

    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()
    

    Y_prediction_tree = dec_tree.predict(X_test_scaled)
    print('tree_predictions ', Y_prediction_tree)

    expected = Y_test
    print('actual_values   ', expected)


    print()
    print('----Tree_report--------------------------------')
    print(classification_report(expected, Y_prediction_tree))

run_tree(X_train_scaled, X_test_scaled, Y_train, Y_test)

def run_svc(X_train_scaled, X_test_scaled, Y_train, Y_test):
    """Run SVC with scikit."""
    # This is where we define the models with pre-defined parameters
    # We can learn these parameters given our data
    print('Defining and fitting models ...')
    t0 = dt.datetime.utcnow()   
    scv = svm.LinearSVC(C=100.)

    scv.fit(X_train_scaled, Y_train)

    joblib.dump(scv, 'models/svc.pickle')

    print('... finished in {} secs.'.format(dt.datetime.utcnow() - t0))
    print()
    

    Y_prediction_svc = scv.predict(X_test_scaled)
    print('tree_predictions ', Y_prediction_svc)

    expected = Y_test
    print('actual_values   ', expected)


    print()
    print('----SVC_report--------------------------------')
    print(classification_report(expected, Y_prediction_svc))

run_svc(X_train_scaled, X_test_scaled, Y_train, Y_test)

def run_random_forest(X_train_scaled, X_test_scaled, Y_train, Y_test):
    """Scikit random forest
    
    Experiment with 'n_estimators'
    """
    
    n_estimators = 30
    
    rf_model = RandomForestClassifier(n_estimators=n_estimators)

    # Train
    clf = clone(rf_model)
    clf = rf_model.fit(X_train_scaled, Y_train)
    
    joblib.dump(clf, 'models/random_forest.pickle')
    
    scores = clf.score(X_train_scaled, Y_train)
    
    
    
    Y_prediction = clf.predict(X_test_scaled)
    print('tree_predictions ', Y_prediction)

    expected = Y_test
    print('actual_values   ', expected)


    print()
    print('----Random forest report--------------------------------')
    print(classification_report(expected, Y_prediction))

run_random_forest(X_train_scaled, X_test_scaled, Y_train, Y_test)

def run_ada_boost(X_train_scaled, X_test_scaled, Y_train, Y_test):
    """Scikit random forest.
    
    For plotting see:
    http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
    
    Experiment with 'n_estimators'
    """
    
    n_estimators = 30
    ada_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                        n_estimators=n_estimators)

    # Train
    clf = clone(ada_classifier)
    clf = ada_classifier.fit(X_train_scaled, Y_train)
    
    joblib.dump(clf, 'models/ada_boost.pickle')
    
    scores = clf.score(X_train_scaled, Y_train)
    
    
    
    Y_prediction = clf.predict(X_test_scaled)
    print('tree_predictions ', Y_prediction)

    expected = Y_test
    print('actual_values   ', expected)


    print()
    print(classification_report(expected, Y_prediction))

run_ada_boost(X_train_scaled, X_test_scaled, Y_train, Y_test)



