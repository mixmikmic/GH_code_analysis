get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("gtd_95to12_0617dist.xlsx")
display(data.head(n=1))

#info about dataset
data.info()

#total no of columns and rows present in data
print "Total no of rows and columns:",data.shape

#Removing columns which has 80% null values
def remove_columns_missing_values(data, min_threshold):
    for col in data.columns:
        rate = data[col].isnull().sum()/float(len(data)) * 100
        if rate >= min_threshold:
            data = data.drop(col,1)
    return data

data = remove_columns_missing_values(data , 80)
print "Total no of features values available :",len(data.columns)

columns_to_drop = ['INT_LOG' , 'INT_MISC', 'INT_ANY', 'INT_IDEO',
                   'eventid','extended','summary', 'scite1' , 'scite2' , 'scite3' , 'dbsource' , 
                   'provstate', 'location',  'city','nwoundte','propextent','nkillter', 
                   'guncertain1', 'nperpcap','nwoundus','nkillus','latitude','longitude',
                   'propcomment', 'weapdetail', 'corp1', 'motive', 'target1']
data = data.drop(columns_to_drop,axis = 1)

#No of columns present after removing columns which has null values more than 80%
data.info()

#Removing columns with redunant,noisy and irrelevant data
columns_to_drop = ['country_txt','region_txt','crit1','crit2','crit3','propextent_txt','weapsubtype1_txt','weaptype1_txt',
                  'natlty1_txt','ransom','nperps','targsubtype1','weapsubtype1','specificity','nwound','nkill','targtype1_txt','targsubtype1_txt','attacktype1_txt']
data = data.drop(columns_to_drop,axis = 1)

#features after removing redunant,noisy and irrelevant data
data.info()

#Total no of null values present in data
print "Total no of null values in data:",data.isnull().values.sum()

#filling null values with median values
features = data.fillna(data.median())

#Total no of null values present in data
features.isnull().values.sum()

#checking unique values in each Gname to find no of terrorist organosation present in data
features["gname"].unique()

#Encoding terrorist organisation with numerical values to train the data
features_new = features.drop("gname" , axis=1)
gname = features["gname"]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(gname)
gname_encoded = le.transform(gname)

#Spliting data into training and testing test to cross validate trained model
#80% of training data and 20% testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_new, 
                                                    gname_encoded, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
important_features = clf.feature_importances_
acc = accuracy_score(y_test , pred)
print acc

X_train_reduced = X_train[X_train.columns.values[(np.argsort(important_features)[::-1])[:9]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(important_features)[::-1])[:9]]]

#Using random forest to train, test and check the accuracy of trained model with reduced feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=100 ,criterion='entropy', random_state=10)
clf = clf.fit(X_train_reduced, y_train)
pred = clf.predict(X_test_reduced)
acc = accuracy_score(y_test , pred)
print acc

