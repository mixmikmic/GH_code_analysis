import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import numpy as np

data = pd.read_csv('Prostate_Cancer.csv',index_col=0)

data.head()

data.info()

new_df=pd.get_dummies(data,columns=['diagnosis_result'],drop_first=True)

new_df.head()

#using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(new_df.drop('diagnosis_result_M',axis=1))
#removing the last column of the dataframe as this table will be used for the feature matrix
scaled_features=scaler.transform(new_df.drop('diagnosis_result_M',axis=1))
new_data=pd.DataFrame(scaled_features,columns=new_df.columns[:-1])

new_data.head()

#feature matrix
X=new_data

#target matrix
y=new_df['diagnosis_result_M']

from sklearn.cross_validation import train_test_split

#allocating 33% of the dataset for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.neighbors import KNeighborsClassifier

# random value of n_neighbors, we will find a better value of k later.
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred=knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred))

#since a precision of only 76%, we'll try to use another value for k

error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate)

knn=KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,y_train)

new_pred=knn.predict(X_test)

print(classification_report(y_test,new_pred))



