# Importing required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
from sklearn import datasets, svm, cross_validation, tree, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib as plt
get_ipython().magic('matplotlib inline')

# Training Data is imported
train = pd.read_csv("./data/train.csv")
train.head()

# Testing Data is imported
test = pd.read_csv("./data/test.csv")
test.head()

# Assigned Integer values to Male and female
genders = {"male": 1, "female": 0}
train["SexF"] = train["Sex"].apply(lambda s: genders.get(s))
test["SexF"] = test["Sex"].apply(lambda s: genders.get(s))

train["SexF"].head()

# Calculating the average number of passengers per Class that survived
titanic = train.groupby('Pclass').mean()
titanic

# Calculating the average number of passengers per Class and Sex that survived
class_sex_grouping = train.groupby(['Pclass','Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot(kind="bar")

# To check the survival count of people of different age groups
group_by_age = pd.cut(train["Age"], np.arange(0, 90, 10))
age_grouping = train.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()

median_ages_train = np.zeros((2,3))

# Calculating the median from the given data
for i in range(0,2):
    for j in range(0,3):
        median_ages_train[i,j] = train[(train['SexF'] == i) & (train['Pclass'] == j+1)]['Age'].dropna().median()
        
median_ages_train

train['AgeFill'] = train['Age']
train[train['Age'].isnull()][['Age', 'AgeFill', 'SexF', 'Pclass']].head(10)
 
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[(train.Age.isnull()) & (train.SexF == i) & (train.Pclass == j+1),'AgeFill'] = median_ages_train[i,j]

# Dropping the unwanted columns and also the NaN values from the Training dataset
train_new = train.drop(['Cabin','Name','Ticket','Sex','Embarked','Age'], axis=1)
train_new = train_new.dropna(axis=0, how='any')
train_new.count()

# Separating the features and labels
X = train_new.drop(['Survived'], axis=1).values
y = train_new['Survived'].values

# For test.csv
median_ages_test = np.zeros((2,3))

# Calculating the median for test data from the given data
for i in range(0,2):
    for j in range(0,3):
        median_ages_test[i,j] = test[(test['SexF'] == i) & (test['Pclass'] == j+1)]['Age'].dropna().median()
        
median_ages_test

test['AgeFill'] = test['Age']
test[test['Age'].isnull()][['Age', 'AgeFill', 'SexF', 'Pclass']].head(10)
 
for i in range(0, 2):
    for j in range(0, 3):
        test.loc[(test.Age.isnull()) & (test.SexF == i) & (test.Pclass == j+1),'AgeFill'] = median_ages_test[i,j]

# Dropping the unwanted columns and also the NaN values from the Test dataset
test_new = test.drop(['Cabin','Name','Ticket','Sex','Embarked','Age'], axis=1)
test_new = test_new.dropna(axis=0, how='any')
test_new.count()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# Decission Tree Classifier
clf_dt = tree.DecisionTreeClassifier(max_depth=10)
clf_dt.fit(X_train, y_train)
clf_dt.score(X_test, y_test)

# AdaBoost Classifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=50)
bdt.fit(X_train, y_train)
bdt.score(X_test, y_test)

# Gradient Boosting Classifier
clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
clf_gb.fit(X_train, y_train)
clf_gb.score(X_test, y_test)

#Random Forest Classifier
clf_rf = ske.RandomForestClassifier(n_estimators=50)
clf_rf.fit(X_train, y_train)
clf_rf.score(X_test, y_test)

# Prediction by Random Forest Classifier
prediction_rf = clf_rf.predict(test_new.values)

# Creating a Dataframe to store the predicted Data
df_output_rf = pd.DataFrame({
    'PassengerId' : test_new.PassengerId.values,
    'Survived': prediction_rf
})

# Writing the output to a csv file
df_output_rf.to_csv("./output/output_rf.csv", sep=',')

# Prediction by Gradient Boosting classifier
prediction_gb = clf_gb.predict(test_new.values)

# Creating a Dataframe to store the predicted Data
df_output_gb = pd.DataFrame({
    'PassengerId' : test_new.PassengerId.values,
    'Survived': prediction_gb
})

# Writing the output to a csv file
df_output_gb.to_csv("./output/output_gb.csv", sep=',')



