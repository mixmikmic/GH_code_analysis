import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

loans = pd.read_csv('loan_data.csv')

loans.info()

loans.head()

loans.describe()

plt.figure(figsize=(10,6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='Credit.Policy=1')
loans[loans['credit.policy'] == 0]['fico'].hist(alpha=0.5, color='red', bins=30, label = 'Credit.Policy=0')
plt.legend()
plt.xlabel('Fico')



plt.figure(figsize=(10,6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha=0.5, color='red', bins=30, label='not.fully.paid=1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha=0.5, color='blue',bins=30, label='not.fully.paid=0')
plt.legend()
plt.xlabel('Fico')



plt.figure(figsize=(10,6))
sns.countplot(x='purpose',data=loans,hue='not.fully.paid')



sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')
plt.xlabel('Fico')
plt.ylabel('int.rate')



plt.figure(figsize=(10,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')

loans.info()

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns = cat_feats, drop_first=True)

final_data.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid',axis=1), final_data['not.fully.paid'],
                                                   test_size=0.3, random_state=101)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

preddtree = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, preddtree))

print(confusion_matrix(y_test, preddtree))

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

pred = rfc.predict(X_test)

print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))

#Random Forest Classifer performed better !!!

