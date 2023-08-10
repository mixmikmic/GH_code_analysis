get_ipython().magic('matplotlib inline')

import time

import numpy as np
import matplotlib.cm as cm

# Standard Python libraries
import os                                    # For accessing operating system functionalities
import json                                  # For encoding and decoding JSON data
import pickle                                # For serializing and de-serializing Python objects

# Libraries that can be pip installed
import requests                              # Simple Python library for HTTP
import pandas as pd                          # Library for building dataframes similar to those in R
import seaborn as sns                        # Statistical visualization library based on Matplotlib
import matplotlib.pyplot as plt  

import psycopg2
from sqlalchemy import create_engine

from sklearn import cross_validation as cv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.datasets.base import Bunch
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import KFold

incident = pd.read_csv(os.path.abspath('C:\project\ship-happens\data\incident.txt'), sep='\t')

incident.head()

print("{} instances with {} features\n".format(*incident.shape))

incident.describe()

incident.fillna(value={'vlength': incident['vlength'].mean()}, inplace=True)
incident.fillna(value={'vdepth': incident['vdepth'].mean()}, inplace=True)
incident.fillna(value={'vessel_age': incident['vessel_age'].mean()}, inplace=True)

incident.describe()

sns.countplot(y='vessel_class', hue='Accident', data=incident)

sns.countplot(y='route_type', hue='Accident', data=incident)

meta = {
    'target_names': list(incident.Accident.unique()),
    'feature_names': list(incident.columns),
    'categorical_features': {
        column: list(incident[column].unique())
        for column in incident.columns
        if incident[column].dtype == 'object'
    },
}

with open('C:\project\ship-happens\data\meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

feature_names = incident[['gross_ton', 'vlength', 'vdepth', 'vessel_class','vessel_age','route_type']]
target_names   = incident['Accident']

splits = cv.train_test_split(feature_names, target_names, test_size=0.2)
X_train, X_test, Y_train, Y_test = splits

def load_data(root='C:\project\ship-happens\data'):
    # Load the meta data from the file 
    with open(os.path.join(root, 'meta.json'), 'r') as f:
        meta = json.load(f) 
    
    names = meta['feature_names']
    
    # Load the readme information 
    with open(os.path.join(root, 'ReadMe.md'), 'r') as f:
        readme = f.read() 
    
    # Remove the target from the categorical features 
    meta['categorical_features'].pop('Accident')
    
    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = X_train, 
        target = Y_train, 
        data_test = X_test, 
        target_test = Y_test, 
        target_names = meta['target_names'],
        feature_names = meta['feature_names'], 
        categorical_features = meta['categorical_features'], 
        DESCR = readme,
    )

mydataset = load_data()

from sklearn.preprocessing import LabelEncoder 

vessel_class = LabelEncoder() 
vessel_class.fit(mydataset.data.vessel_class)
print(vessel_class.classes_)

from sklearn.base import BaseEstimator, TransformerMixin

class EncodeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None. 
    """
    
    def __init__(self, columns=None):
        self.columns  = [col for col in columns] 
        self.encoders = None
    
    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to encode. 
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns 
        
        # Fit a label encoder for each column in the data frame
        self.encoders = {
            column: LabelEncoder().fit(data[column])
            for column in self.columns 
        }
        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame. 
        """
        output = data.copy()
        for column, encoder in self.encoders.items():
            output[column] = encoder.transform(data[column])
            
        return output

encoder = EncodeCategorical(mydataset.categorical_features.keys())
data = encoder.fit_transform(mydataset.data)

from sklearn.preprocessing import Imputer 

class ImputeCategorical(BaseEstimator, TransformerMixin):
    """
    Encodes a specified list of columns or all columns if None. 
    """
    
    def __init__(self, columns=None):
        self.columns = columns 
        self.imputer = None
    
    def fit(self, data, target=None):
        """
        Expects a data frame with named columns to impute. 
        """
        # Encode all columns if columns is None
        if self.columns is None:
            self.columns = data.columns 
        
        # Fit an imputer for each column in the data frame
        self.imputer = Imputer(missing_values=0, strategy='most_frequent')
        self.imputer.fit(data[self.columns])

        return self

    def transform(self, data):
        """
        Uses the encoders to transform a data frame. 
        """
        output = data.copy()
        output[self.columns] = self.imputer.transform(output[self.columns])
        
        return output

    
imputer = ImputeCategorical(['vessel_class', 'route_type'])
data = imputer.fit_transform(data)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# we need to encode our target data as well. 
yencode = LabelEncoder().fit(mydataset.target)

# construct the pipeline 
incident = Pipeline([
        ('encoder',  EncodeCategorical(mydataset.categorical_features.keys())),
        ('imputer', ImputeCategorical(['vessel_class', 'route_type'])), 
        ('classifier', LogisticRegression())
    ])

# fit the pipeline 
incident.fit(mydataset.data, yencode.transform(mydataset.target))

from sklearn.metrics import classification_report 

# encode test targets, and strip traililng '.' 
y_true = yencode.transform([y.rstrip(".") for y in mydataset.target_test])

# use the model to get the predicted value
y_pred = incident.predict(mydataset.data_test)

# execute classification report 
print(classification_report(y_true, y_pred, target_names=mydataset.target_names))

#print(mydataset)

import pickle 

def dump_model(model, path='C:\project\ship-happens\output', name='incident.pickle'):
    with open(os.path.join(path, name), 'wb') as f:
        pickle.dump(model, f)
        
dump_model(incident)

def load_model(path='C:\project\ship-happens\output/incident.pickle'):
    with open(path, 'rb') as f:
        return pickle.load(f) 


def predict(model, meta=meta):
    data = {} # Store the input from the user
    
    for column in meta['feature_names'][:-1]:
        # Get the valid responses
        valid = meta['categorical_features'].get(column)
    
        # Prompt the user for an answer until good 
        while True:
            val = "" + input("enter {} >".format(column))
            if valid and val not in valid:
                print("Not valid, choose one of {}".format(valid))
            else:
                data[column] = val
                break
    
    # Create prediction and label 
    yhat = model.predict(pd.DataFrame([data]))
    return yencode.inverse_transform(yhat)
            
    
# Execute the interface 
model = load_model()
predict(model)



