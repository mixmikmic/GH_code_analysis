import pandas as pd
import numpy as np

emp_data = pd.read_csv('HR_comma_sep.csv')

emp_data.head()

emp_data.columns

emp_data.info()

columns = emp_data.columns.tolist()

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import itertools

categorical  = ['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','sales','salary']

fig=plt.subplots(figsize=(10,15))
length = len(categorical)
for i,j in itertools.zip_longest(categorical,range(length)):
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=0.5)
    sns.countplot(x=i, data= emp_data)
    
plt.subplot(np.ceil(length/2),2,6)
plt.xticks(rotation=90)
    

l = zip(categorical,range(6))

next(l)

next(l)

emp_data.shape

categorical  = ['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary']

fig=plt.subplots(figsize=(12,15))

length = len(categorical)

for i,j in itertools.zip_longest(categorical,range(length)):
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    #Based on left colum, splits the data
    sns.countplot(x=i,data = emp_data, hue="left")
    plt.xticks(rotation=90)

# here we will do it only for categorical variable
categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','sales','salary']
length = len(categorical)

fig=plt.subplots(figsize=(12,15))
for i,j in itertools.zip_longest(categorical,range(length)):
    # only counting the number who left 
    Proportion_of_data = emp_data.groupby([i])['left'].agg(lambda x: (x==1).sum()).reset_index()
    
    # Counting the total number 
    Proportion_of_data1=emp_data.groupby([i])['left'].count().reset_index() 
    
    # mergeing two data frames
    Proportion_of_data2 = pd.merge(Proportion_of_data,Proportion_of_data1,on=i) 
    
    # Now we will calculate the % of employee who left category wise
    Proportion_of_data2["Proportion"]=(Proportion_of_data2['left_x']/Proportion_of_data2['left_y'])*100 
    
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    sns.barplot(x=i,y='Proportion',data=Proportion_of_data2)
    plt.xticks(rotation=90)
    plt.title("percentage of employee who left")
    plt.ylabel('Percentage')

emp_data.groupby(['number_project'])['left'].agg(lambda x: (x==1).sum())

emp_data.groupby(['number_project'])['left'].count()

corr = emp_data.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,cbar=True)
plt.xticks(rotation=90)

# For changing categorical variable into int
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
emp_data['salary'] = le.fit_transform(emp_data['salary'])
emp_data['sales'] = le.fit_transform(emp_data['sales'])

# we can select importance features by using Randomforest Classifier
from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100)

#Using all features except left
feature_var = emp_data.loc[:,emp_data.columns != "left"]

#Getting target features
pred_var = emp_data.loc[:,emp_data.columns == "left"]



pred_var.values.ravel()

model.fit(feature_var,pred_var.values.ravel())

model.feature_importances_

feature_var.columns

pd.Series(model.feature_importances_,index=feature_var.columns).sort_values(ascending=False)

# Importing Machine learning models library used for classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

def Classification_model(model,Data,x,y): # here x is the variable which are used for prediction
    # y is the prediction variable
    train,test = train_test_split(Data,test_size= 0.33)
    train_x = Data.loc[train.index,x] # Data for training only with features
    train_y = Data.loc[train.index,y] # Data for training only with predcition variable
    test_x = Data.loc[test.index,x] # same as for training 
    test_y = Data.loc[test.index,y]
    
    model.fit(train_x,train_y.values.ravel())
    
    pred=model.predict(test_x)
    
    accuracy=accuracy_score(test_y,pred)
    return accuracy

All_features=['satisfaction_level',
'number_project',
'time_spend_company',
'average_montly_hours',
'last_evaluation',
'sales',
'salary',
'Work_accident',       
'promotion_last_5years']

Important_features = ['satisfaction_level',
'number_project',
'time_spend_company',
'average_montly_hours',
'last_evaluation']

#Target Variable
Pred_var = ["left"]

models=["RandomForestClassifier","Gaussian Naive Bays","KNN","Logistic_Regression","Support_Vector"]

Classification_models = [RandomForestClassifier(n_estimators=100),GB(),knn(n_neighbors=7),LogisticRegression(),SVC()]

Model_Accuracy = []

for model in Classification_models:
    Accuracy=Classification_model(model,emp_data,Important_features,Pred_var)
    Model_Accuracy.append(Accuracy)

Model_Accuracy

Model_Accuracy = []
for model in Classification_models:
    Accuracy=Classification_model(model,emp_data,All_features,Pred_var)
    Model_Accuracy.append(Accuracy)

Model_Accuracy



