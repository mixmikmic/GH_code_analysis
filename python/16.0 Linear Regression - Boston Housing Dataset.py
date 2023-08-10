import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_boston

boston=load_boston()

boston.keys()

boston['data'][1:3]

boston['target'][1:10]

boston['feature_names'][:]

print(boston['DESCR'])

boston_data = np.array( boston['data']) 

boston_data_pd = pd.DataFrame(data= boston['data'], columns = boston['feature_names'])

boston_data_pd.head()

boston_data_pd.info()

boston_data_pd.describe()

boston_data_pd.columns[:]

sns.pairplot(boston_data_pd)

sns.distplot(boston_data_pd['CRIM'])

sns.distplot(boston_data_pd['ZN'])

sns.distplot(boston_data_pd['INDUS'])

sns.distplot(boston_data_pd['CHAS'])

sns.distplot(boston_data_pd['NOX'])

sns.distplot(boston_data_pd['RM'])

sns.distplot(boston_data_pd['AGE'])

sns.distplot(boston_data_pd['DIS'])

sns.distplot(boston_data_pd['RAD'])

sns.distplot(boston_data_pd['TAX'])

sns.distplot(boston_data_pd['PTRATIO'])

sns.distplot(boston_data_pd['B'])

X = boston_data_pd
#df.drop(    'Price',axis=1)
#X = df.drop('Address',axis=1)

X.head(2)

Y = pd.DataFrame(boston['target'])
Y.head(2)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,Y_train)

lm.intercept_

lm.coef_[0]

coef = pd.DataFrame(lm.coef_[0],X.columns,columns=['Coeff'])
                    

coef

prediction= lm.predict(X_test)

plt.scatter(Y_test,prediction)

sns.distplot((Y_test-prediction))

from sklearn import metrics 

matx = metrics.mean_absolute_error(Y_test,prediction) 

np.sqrt(matx)



