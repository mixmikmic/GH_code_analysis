import math
import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

df = pd.read_csv('03 WA_Fn-UseC_-Marketing-Campaign-Eff-UseC_-FastF.csv')
df.drop(['LocationID','AgeOfStore','week'],axis=1,inplace=True)
df.info()

temp = df.pivot_table(values=['SalesInThousands'], index=['MarketID'], columns=['Promotion'], aggfunc='sum')
temp.plot(figsize=(8,4))

temp = df.pivot_table(values=['SalesInThousands'], index=['MarketSize'], columns=['Promotion'], aggfunc='sum')
temp.plot(figsize=(8,4))

cat_feats = ['MarketID','MarketSize','Promotion']
final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)

x = final_data.drop(['SalesInThousands'],axis=1)
y = final_data['SalesInThousands']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=101)

x_train = sm.add_constant(x_train).copy()
lm_sm = sm.OLS(y_train, x_train)
est = lm_sm.fit()
print(est.summary())

sns.heatmap(x_train.corr(),cmap='coolwarm',annot=True)

x_test = sm.add_constant(x_test).copy()
y_pred = est.predict(x_test)

print('BIAS:', round(np.mean(y_test - y_pred),2))
print('MAPE:', round(np.mean(np.absolute((y_test - y_pred)/y_test))*100,2), "%")

plt.scatter(y_test,y_pred,color='darkred')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

