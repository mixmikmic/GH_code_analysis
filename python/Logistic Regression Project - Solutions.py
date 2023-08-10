import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

ad_data = pd.read_csv('advertising.csv')

ad_data.head()



ad_data.describe()



sns.set_style('whitegrid')

ad_data['Age'].hist(bins = 30)

sns.jointplot('Age', 'Area Income', data = ad_data, kind = 'hex')

sns.jointplot('Age', 'Area Income', data = ad_data)

sns.jointplot(y= 'Daily Time Spent on Site', x = 'Age', data = ad_data, kind = 'kde', color = 'red')



sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data = ad_data, color = 'green')



sns.pairplot(ad_data, hue = 'Clicked on Ad', palette = 'bwr')

from sklearn.cross_validation import train_test_split
ad_data.columns

X = ad_data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp', 'Clicked on Ad'], axis = 1)
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)



predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))



