import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv("data/hr/human-resources-analytics.zip",compression='zip')

len(df)

len(df.columns)

df.dtypes

df.isnull().sum()*100/len(df)

df.describe(include='all')

df.head()

df.sales.value_counts()

df.left.value_counts()*100/len(df)

df.salary.value_counts()/len(df)

df1 = df[df['left']==0]
df1.salary.value_counts()*100/len(df1)

df1 = df[df['left']==1]
df1.salary.value_counts()*100/len(df1)

salary_map = {'low':1,'medium':2,'high':3}
df['salary_int'] = df.salary.map(salary_map)

df['number_project'].value_counts().sort_index().plot(kind='bar')

df['satisfaction_level'].plot.kde()

# box plot of satisfaction_level for each department
df[['sales','satisfaction_level']].boxplot(by='sales',figsize=(16,8))

# heat map of correlation matrix
sns.heatmap(df.corr(),annot=True,cmap='RdYlGn_r', linewidths=0.5)

df['colors'] = df.sales.apply(lambda x: x.lower()[0])

df.columns

#scatter plot of satisfaction_level and last_evaluation for differt departments
# change the hue variables to see how the data is distributed

fg = sns.FacetGrid(data=df, hue='sales', aspect=1.61,size=10,
                   hue_kws={'s':df.salary_int*50})
fg.map(plt.scatter, 'satisfaction_level', 'last_evaluation').add_legend()

cols = ['sales','satisfaction_level']
df1 = df[df['left']==1]
df_grp = df1[cols].groupby(['sales'])
dx = df_grp.agg([len,np.sum,np.max,np.min,np.mean,np.std])
dx.columns = pd.MultiIndex.droplevel(dx.columns,0)
dx['pct_sat'] = dx['sum']*100/dx['len'].sum()
dx['left'] = 1
dx

cols = ['sales','satisfaction_level']
df1 = df[df['left']==0]
df_grp = df1[cols].groupby(['sales'])
dy = df_grp.agg([len,np.sum,np.max,np.min,np.mean,np.std])
dy.columns = pd.MultiIndex.droplevel(dy.columns,0)
dy['pct_sat'] = dy['sum']*100/dy['len'].sum()
dy['left'] = 0
dy

dz = pd.concat([dx,dy],axis=1)
dz.sort_index()

#df.pivot_table(aggfunc=len,index=['sales'])
#df.pivot_table(aggfunc=len,index=['sales'],columns=['left'],values=['satisfaction_level'])
df.pivot_table(aggfunc=np.mean,index=['sales'],columns=['salary','left'],values=['satisfaction_level'])

df_final = df.drop(['salary'],axis=1).copy()

df_final.sales.value_counts()

pd.get_dummies(df_final,columns=['sales']).head()

df_final = pd.get_dummies(df_final,columns=['sales'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
y = df_final['left']
X = df_final.drop(['left'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

my_list = list(zip(list(y_test),list(y_pred)))

wronng_cls = [pair for pair in list(enumerate(my_list)) if pair[1][0] != pair[1][1]]
wronng_cls[0:10]

len(X_test),len(wronng_cls)

1 - (1058.0/4950)

X_test.iloc[102,:]

accuracy_score(y_test,y_pred)

features = df_final.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression()

clf_lr.fit(X_train,y_train)

y_pred = clf_lr.predict(X_test)

accuracy_score(y_test,y_pred)



