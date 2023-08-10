import pandas as pd # begin if read data from three files
import numpy as np

# read 3 part features of predicting data by 3 different types to save memory
X=pd.read_csv('./Xint8_predictlevel123.dat', sep = '\t', dtype=np.int8)
Xint16=pd.read_csv('./Xint16_predictlevel123.dat', sep = '\t', dtype=np.int16)
X_gender=pd.read_csv('./X_gender_predictlevel123.dat', sep = '\t')

# read header of int16 type features
with open('headers123.small.txt') as f:
    list16 = f.read().splitlines()

# add features from another two files to X
X[list16]=Xint16[list16]
X['gender']=X_gender['gender']
X

# read the Patient_ID column to write into result file (used only when read X from 3 files)
df_training_data = pd.read_csv('./trn_final_predict_with15.txt', sep = '\t')
y_patient_col = df_training_data[['Patient_ID','Buy_Diabetes']]

X.info()

#change X to dictionary and vectorize it

import sys
import os
import pickle
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from itertools import tee, islice, chain
get_ipython().magic('matplotlib inline')


X = X.T.to_dict().values()


# Verctirize X to vectors
# read vectorizer create by trainer to make sure has same number of features
# here use transform() instead of fix_transform() which used in trainer
with open('./vectorizer.dat', 'rb') as vectorizer_f:
    vectorizer = pickle.load(vectorizer_f)
X = vectorizer.transform(X)
print(X)

# load the saved classifier
classifier_path = './classifier.dat'
with open(classifier_path, 'rb') as classifier_f:
    classifier = pickle.load(classifier_f)
print('classifier: ',classifier)

#  predict result       
#predict_y = classifier.predict(X.toarray()) # get 0/1
# get probability
predict_y = classifier.predict_proba(X.toarray())
predict_y = predict_y[:,1]

print(predict_y)
print(type(predict_y))

# get the format final result file need
y_patient_col[['Patient_ID','Buy_Diabetes']] = df_training_data[['Patient_ID','Buy_Diabetes']]
print(type(y_patient_col))
print(predict_y)
y_patient_col['Diabetes'] = predict_y.tolist()
print(y_patient_col)
y_patient_col = y_patient_col[['Patient_ID','Diabetes']]
y_patient_col


# write to file
path_to_results = './prob_dbgbmatc123_cout.csv'
y_patient_col.to_csv(path_to_results,index=False)

# read data

import pandas as pd
import os
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np

# datas have been drop duplicate and collected in SQL Server
# except ATClevelcode part, since pandas pivot_table is a better tool to deal with it

# read features of training dataset except features about ATClevelcode
df_training_data = pd.read_csv('./trn_final_predict_with15.txt', sep = '\t') #change

# read ATClevelcode(level 1-3) associated data from training dataset
# only (Patient_ID,year_atclevel,cnt) columns, need to be aggregated in next step
df_atc_data = pd.read_csv('./trn_predicting_atc123.txt', sep = '\t') #change

# read the Patient_ID column to write into result file  
y_patient_col = df_training_data[['Patient_ID','Buy_Diabetes']]

df_atc_data

df_training_data

# pivote table, the values of year_atclevel as column name
# year_atlevel including the combination of (2011-2014,2015Q1-2015Q4) and (all ATC level 1-3 codes)
pv_table = pd.pivot_table(df_atc_data,values='cnt',index=['Patient_ID'],
                          columns=['year_atclevel'],aggfunc='sum').fillna(0).astype(int)
X=pd.DataFrame(pv_table.to_records())
X

# In order to make predicting data matrix has the same columns as training data
# add some records using Patient_ID=0, delete this virtual patient after pivote table
X = X[X['Patient_ID']!=0]
# some columns in predicting data matrix has values need int16, but only use int8 in training data
# set the values as max of int8 (only 5 columns is ok in more than two thousand columns)
X.loc[X['2011N02C']>=255,'2011N02C']=255
X.loc[X['2012N02C']>=255,'2012N02C']=255
X.loc[X['2012N05']>=255,'2012N05']=255
X.loc[X['2012N05B']>=255,'2012N05B']=255
X.loc[X['2013N05B']>=255,'2013N05B']=255

# make sure index begin from 0, so that the atc features index is same as other features 
X=X.reset_index()
X

del X['index']
X

# add features together
X[['gender','year_of_birth','postcode','state_code','trans_all','trans_lipids','trans_hypertension','trans_Depression','trans_Immunology','trans_Urology','trans_Anti_Coagulant','trans_Osteoporosis','trans_Heart_Failure','trans_Epilepsy','trans_COPD','trans_Diabetes','Buy_Diabetes','dur_Diabetes_15']]=df_training_data[['gender','year_of_birth','postcode','state_code','trans_all','trans_lipids','trans_hypertension','trans_Depression','trans_Immunology','trans_Urology','trans_Anti_Coagulant','trans_Osteoporosis','trans_Heart_Failure','trans_Epilepsy','trans_COPD','trans_Diabetes','Buy_Diabetes','dur_Diabetes_15']]
X.loc[X['dur_Diabetes_15']>0, 'dur_Diabetes_15']=1
X

with open('headers123.small.txt') as f: # headers123.small.txt has all the feature names of int16
    list16 = f.read().splitlines()
Xint16=X[list16]
Xint16.to_csv('./Xint16_predictlevel123.dat', sep='\t', index=False)

with open('headers123.txt') as f:
    list8 = f.read().splitlines()
Xint8=X[list8]
Xint8.to_csv('./Xint8_predictlevel123.dat', sep='\t', index=False)

X_gender=df_training_data[['Patient_ID','gender']]
X_gender.to_csv('./X_gender_predictlevel123.dat', sep='\t', index=False)

Xint16

