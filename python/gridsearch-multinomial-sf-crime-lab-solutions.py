import numpy as np
import pandas as pd
import patsy

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

crime_csv = '../datasets/sf_crime_train.csv'

sf_crime = pd.read_csv(crime_csv)
sf_crime.head()

# There is a column that is is a Datetime and I want to check and see if it is currently an object
sf_crime.Dates.dtype

# pd.datetime was not helpful
sf_time = pd.DataFrame(sf_crime['Dates'].str.split(' ',1).tolist(),columns = ['date','time'])
# sf_time is a dataframe where the Date and time are in separate columns

sf_date = pd.DataFrame(sf_time['date'].str.split('/').tolist(),columns = ['month','day','year'])
# sf_date is a dataframe where all the month, day and year are all in separate columns

# Merge data frames with individual time values back onto main df
sf_crime = sf_crime.merge(sf_date, left_index = True, right_index = True,how = 'outer')
sf_crime = sf_crime.merge(sf_time, left_index = True, right_index = True,how = 'outer')

# Check out Current dataframe if you are interested
sf_crime.head(2)

# Dropping colums where time is expressed in human speak
sf_crime.drop(['Dates','date'], axis = 1, inplace = True)

sf_crime['Category'].value_counts()

# 1 Trespassing, all others are trespass,  1 Assualt because someone can't spell.

sf_crime['DayOfWeek'].value_counts()
# all days of week are there

sf_crime['PdDistrict'].value_counts()
# Values look good

sf_crime['Resolution'].value_counts()
# 1 non prosecuted.  Seems legit

sf_crime[['X','Y']].describe()
# all coordinates appear to be legitimate

# Figuring out where that wrong data exists in the DataFrame
sf_crime[sf_crime['Category'] == 'ASSUALT']
# rows 2750 and 4330

sf_crime[sf_crime['Category'] == 'TRESPASSING']
# row 5519

# Issues with data are small enough to be manually changed
sf_crime.set_value(2750, 'Category', 'ASSAULT')
sf_crime.set_value(4330, 'Category', 'ASSAULT')
sf_crime.set_value(5519, 'Category', 'TRESPASS')

# Multiclass regression.  Our Baseline will probably be Violent as that is the left-over group so to speak.
# However, it would naturally make sence to set our baseline be the class that has the most observations.

#First i'll need to convert sub categories into overlaying categories.

zeros = ['non-criminal', 'runaway', 'secondary codes', 'suspicious occ', 'warrants']
ones  = ['bad checks', 'bribery', 'drug/narcotic', 'drunkenness', 'embezzlement', 'forgery/counterfeiting', 'fraud', 
         'gambling','liquor', 'loitering', 'trespass', 'other offenses']
#twos  = all other things  

# Empty list to append values into
crime_cat = []
#iterate through sf_crime Category
for crime in sf_crime['Category']:
    # convert values to lower
    crime = crime.lower()
    # checks list of sub categories
    if crime in zeros:
        # appends the overlaying category
        crime_cat.append('non-crime')
    elif crime in ones:
        crime_cat.append('non-violent')
    else:
        crime_cat.append('violent')
        
# take that list and add it to the DF
sf_crime['cat_number'] = crime_cat

# also going to convert DayOfWeek, PdDistrict and Resolution to dummy variables.
dummies = pd.get_dummies(sf_crime[['DayOfWeek','PdDistrict','Resolution']], drop_first = True)

# Merge the dataframe result back onto the original dataframe
sf_crime = sf_crime.merge(dummies, left_index = True, right_index = True,how = 'outer')

# Dropping all the categorical values the I don't think will be relevant or have been converted to dummies for X
X = sf_crime.drop(['Category','Descript','DayOfWeek','PdDistrict',
                   'Resolution','Address','X','Y','cat_number','time'], axis = 1)
y = sf_crime['cat_number'].values

y

X.columns

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
Xs = ss.fit_transform(X)

# Example:
# fit model with five folds and lasso regularization
# use Cs=15 to test a grid of 15 distinct parameters
# remember: Cs describes the inverse of regularization strength

# logreg_cv = LogisticRegressionCV(solver='liblinear', 
#                                  Cs=[1,5,10], 
#                                  cv=5, penalty='l1')

# TTS our data.
# We will have a holdout set to test on at the end.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

y_test

# Lets set our model parameters 
logreg_cv = LogisticRegressionCV(Cs=100, cv=5, penalty='l1', scoring='accuracy', solver='liblinear')
logreg_cv.fit(X_train, y_train)

# find best C per class  
# Building a dictionary that does a regression for each of the Y classes
# after the fit it grabs the C value for said logistic regression and puts them together.
best_C = {logreg_cv.classes_[i]:x for i, (x, c) in enumerate(zip(logreg_cv.C_, logreg_cv.classes_))}
print('best C for class:', best_C)

# fit regular logit model to 'non-crime', 'non-violent', and 'violent' classes
# use lasso penalty
logreg_1 = LogisticRegression(C=best_C['non-crime'], penalty='l1', solver='liblinear', multi_class = 'ovr')
logreg_2 = LogisticRegression(C=best_C['non-violent'], penalty='l1', solver='liblinear', multi_class = 'ovr')
logreg_3 = LogisticRegression(C=best_C['violent'], penalty='l1', solver='liblinear', multi_class = 'ovr')

# Lets check out all of our outputs for all of our models
# fit model for predicting Non Crimes
logreg_1.fit(X_train, y_train)

# fit model for predicting Non Violent
logreg_2.fit(X_train, y_train)

# fit model for predicting Violent
logreg_3.fit(X_train, y_train)

# using our logregs to predict on our test set and storing predictions
Y_1_pred = logreg_1.predict(X_test)
Y_2_pred = logreg_2.predict(X_test)
Y_3_pred = logreg_3.predict(X_test)

# stores confusion matrix for Y Test and Y Pred  
conmat_1 = confusion_matrix(y_test, Y_1_pred, labels=logreg_1.classes_)
# converts np.matrix format matrix to a dataframe and adds index and column names
conmat_1 = pd.DataFrame(conmat_1, columns=logreg_1.classes_, index=logreg_1.classes_)

conmat_2 = confusion_matrix(y_test, Y_2_pred, labels=logreg_2.classes_)
conmat_2 = pd.DataFrame(conmat_2, columns=logreg_2.classes_, index=logreg_2.classes_)

conmat_3 = confusion_matrix(y_test, Y_3_pred, labels=logreg_3.classes_)
conmat_3 = pd.DataFrame(conmat_3, columns=logreg_3.classes_, index=logreg_3.classes_)

print 'best params for non-crime:'
print conmat_1
print 'best params for non-violent:'
print conmat_2
print 'best params for violent:'
print conmat_3

print(classification_report(y_test, Y_1_pred))
print(classification_report(y_test, Y_2_pred))
print(classification_report(y_test, Y_3_pred))

# Precision ( True Positives divided by Total Predicted Positives)

# Recall (True Positives divided by Total Actual Positives)

# f1-score ( 2 * (precision * recall) / (precision + recall) )

# Support -  Number of True Values in said class

# We can observe subtle differences in the confusion matrix, 
# but overall our classification reports are identical.  

# this leads us to believe that there variables that are highly
# predictive of specific classes.

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

gs_params = {
    'penalty':['l1','l2'],
    'solver':['liblinear'],
    'C':np.logspace(-5,0,100)
}

lr_gridsearch = GridSearchCV(LogisticRegression(), gs_params, cv=5, verbose=1, n_jobs=-1)

lr_gridsearch.fit(X_train, y_train)



