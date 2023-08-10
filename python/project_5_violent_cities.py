import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
import dateutil.parser
import datetime
datetime
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

df= pd.read_excel('data.xlsx')

df.head()

df_location= df['City'].apply(lambda x: pd.Series(x.split(',')))

df_location.tail()

df_final= pd.concat([df, df_location], axis=1)
df_final.columns

df_final.columns = ['Rank', 'nothing','Murder Rate (per 100,000)', 'City', 'Country']

df_final.head(7)

df_final.value_counts()

df_final.to_csv('project_5_PythonExport.csv', sep=',')

import seaborn as sns
test_df = pd.melt(df_final, id_vars=['Country', 'City'], value_vars=['Murder Rate (per 100,000)'])
ax = sns.stripplot(x='value', y='Country', data=test_df)
plt.savefig("project5.pdf", transparent=True)

fig, ax = plt.subplots()
df_final['Country'].value_counts().plot(kind= 'bar', stacked= True)
#plt.savefig("project5.pdf", transparent=True)

fig, ax = plt.subplots()
df_final.plot(kind= 'bar', x='City', y= 'Murder Rate (per 100,000)', figsize= (12,5), ax=ax)

df_venezuela = df_final[(df_final['Country'] == ' Venezuela')]
df_venezuela

fig, ax = plt.subplots()
df_venezuela.plot(kind= 'bar', x='City', y= 'Murder Rate (per 100,000)', figsize= (12,5), ax=ax)

plt.scatter(df_final['City'], df_final['Murder Rate (per 100,000)'], edgecolor='none', c= df['Country'], alpha= 0.5)



