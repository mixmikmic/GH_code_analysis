import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

get_ipython().magic('matplotlib inline')

ad_data = pd.read_csv('advertising.csv')

ad_data.head()



ad_data.info()





ad_data.describe()

plt.hist(ad_data['Age'])





sns.jointplot(data=ad_data, x='Age', y='Area Income')



sns.jointplot(data=ad_data, x='Age', y='Daily Internet Usage', kind='kde', color='#ffe468')



sns.jointplot(data=ad_data, x='Daily Time Spent on Site', y='Daily Internet Usage')



sns.pairplot(data=ad_data, hue='Clicked on Ad')



X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]

y = ad_data['Clicked on Ad']

x_train, x_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print confusion_matrix(y_test, y_pred)

print accuracy_score(y_test, y_pred)

print classification_report(y_test, y_pred)





