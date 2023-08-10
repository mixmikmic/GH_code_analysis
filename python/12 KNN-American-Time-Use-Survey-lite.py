import math
import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('12 WA_American-Time-Use-Survey-lite.csv')
df.info()

df1 = df[(df['Age']>=20) & (df['Age']<=69)].copy()
df2 = df1[(df1['Weekly Hours Worked']<=69) & (df1['Weekly Hours Worked']>=10)].copy()
df3 = df2[(df2['Weekly Earnings']>200) & (df2['Weekly Earnings']<2000)].copy()
df4 = df3[df3['Children']<=3].copy()

def func_EL (x):
    if x in ('9th grade','10th grade','11th grade','12th grade','High School'): return '1. Lower'
    elif x in ('Some College','Associate Degree'): return '2. Meduim'
    elif x in ('Bachelor','Master'): return '3. Higher'
    elif x in ('Doctoral Degree','Prof. Degree'): return '4. Professional'
    else: return '5. Others'
df4['Education Level bin'] = df4['Education Level'].apply(func_EL)

def func_WHW (x):
    if x<25: return '1. y < 25'
    elif x<40: return '2. y < 40'
    elif x==40: return '3. y = 40'
    elif x<50: return '4. y < 50'
    else: return '5. y >= 50'
df4['Weekly Hours Worked bin'] = df4['Weekly Hours Worked'].apply(func_WHW)

df = df4[['Education Level bin','Gender','Weekly Earnings','Weekly Hours Worked bin']].copy()
df.dropna(inplace=True)
df.info()

temp = pd.DataFrame(df.groupby(['Education Level bin'], axis=0, as_index=False)['Weekly Earnings'].mean())
plt.figure(figsize=(8,4))
sns.barplot(x="Education Level bin", y="Weekly Earnings",data=temp)

temp = pd.DataFrame(df.groupby(['Weekly Hours Worked bin'], axis=0, as_index=False)['Weekly Earnings'].mean())
plt.figure(figsize=(8,4))
sns.barplot(x="Weekly Hours Worked bin", y="Weekly Earnings",data=temp)

def WE_bins (x):
    if x <= 500: return "a. less than 500"
    else: return "b. more than 500"
df['actual'] = df['Weekly Earnings'].apply(WE_bins)

cat_feats = ['Education Level bin','Gender','Weekly Hours Worked bin']
final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)

x = final_data.drop(['Weekly Earnings','actual'],axis=1)
y = final_data['actual']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=101)

error_rate = []
for i in range(1,51,5):    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(8,4))
plt.plot(range(1,51,5),error_rate,color='darkred', marker='o',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print('\033[1m' + '\033[4m' + 'accuracy' + '\033[0m')
print(round(np.mean(y_test==y_pred)*100,2),"%")
print('\n')
print('\033[1m' + '\033[4m' + 'classification_report' + '\033[0m')
print(classification_report(y_test,y_pred))
print('\n')
print('\033[1m' + '\033[4m' + 'confusion_matrix' + '\033[0m')
print(confusion_matrix(y_test,y_pred))



