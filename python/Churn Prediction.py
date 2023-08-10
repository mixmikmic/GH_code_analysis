import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

train_data.head()

#train_data.info()

null_data = train_data.isnull()

plt.figure(figsize=(12,6))
sns.heatmap(null_data, cmap='viridis')

train_data['Network type subscription in Month 1'].fillna('2G', inplace = True)

train_data['Network type subscription in Month 2'].fillna('3G', inplace = True)

train_data.dropna(inplace = True)

plt.figure(figsize=(12,6))
sns.heatmap(train_data.isnull(), cmap='viridis')



train_data.drop(['Customer tenure in month', 'Most Loved Competitor network in in Month 1', 'Most Loved Competitor network in in Month 2', 'Network type subscription in Month 1', 'Network type subscription in Month 2', 'Customer ID'], axis = 1, inplace = True)

#train_data = train_data.iloc[:600]

#corr_matrix = train_data.corr()
#f, ax = plt.subplots(figsize = (18, 10))
#sns.heatmap(corr_matrix, linewidths = 2.0, annot = True)
#ax.set_title('Correlation matrix')

test_data.drop(['Customer tenure in month', 'Most Loved Competitor network in in Month 1', 'Most Loved Competitor network in in Month 2', 'Network type subscription in Month 1', 'Network type subscription in Month 2', 'Customer ID'], axis = 1, inplace = True)

test_data.head()

#test_data.info()

y = train_data['Churn Status']
x = train_data.drop(['Churn Status'], axis = 1, inplace = True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x = train_data.as_matrix().astype(np.float)
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 300)
rfc.fit(x_train, y_train)

rfc_predictions = rfc.predict(test_data)

#rfc_predictions

from sklearn.metrics import classification_report,confusion_matrix

#rfc_predictions

print(classification_report(y_test, rfc_predictions))

rfc_predictions = np.asarray(rfc_predictions, dtype = int)

train_data = pd.read_csv('Test.csv')
customer_id = train_data['Customer ID']

submision = pd.DataFrame(rfc_predictions, columns = ['Churn Status'], index = customer_id )

#submision.to_csv('final output1.csv')

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(x_train, y_train)

dtree_predictions = dtree.predict(test_data)

print(classification_report(y, dtree_predictions))



