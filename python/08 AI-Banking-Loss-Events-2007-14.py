import math
import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')

df = pd.read_csv('08 WA_Fn-UseC_-Banking-Loss-Events-2007-14.csv')
df.info()

df['Net_Loss']=df['Net Loss'].apply(lambda x: float(x.replace(',', '')))
df['Gross_Loss']=df['Estimated Gross Loss'].apply(lambda x: float(x.replace(',', '')))
df = df[['Region','Business','Risk Category','Year','Net_Loss','Gross_Loss']].copy()
df.info()

temp = pd.DataFrame(df.groupby(['Year'], axis=0, as_index=False)['Net_Loss'].sum())
plt.figure(figsize=(8,4))
sns.pointplot(x="Year", y="Net_Loss",data=temp)

temp = pd.DataFrame(df.groupby(['Region'], axis=0, as_index=False)['Net_Loss'].sum())
plt.figure(figsize=(8,4))
sns.barplot(x="Region", y="Net_Loss",data=temp)

def func_region(x):
    if x=="Corporate Finance": return "a. Corporate Finance"
    if x=="Retail Banking": return "b. Retail Banking"
    elif x=="Trading and Sales": return "c. Trading and Sales"
    elif x=="Commercial Banking": return "d. Commercial Banking"
    else: return "e. Others"
df['Business bin'] = df['Business'].apply(func_region)

temp = pd.DataFrame(df.groupby(['Business bin'], axis=0, as_index=False)['Net_Loss'].sum())
plt.figure(figsize=(8,4))
sns.barplot(x="Business bin", y="Net_Loss",data=temp)
plt.tight_layout()

def func_riskcat (x):
    if x=="Execution, Delivery and Process Management": return "a. Process Management"
    elif x=="External Fraud": return "b. External Fraud"
    elif x=="Clients, Products and Business Practices": return "c. Business Practices"
    elif x=="Employment Practices and Workplace Safety": return "d. Employment Practices"
    else: return "e. Others"
df['Risk Category bin'] = df['Risk Category'].apply(func_riskcat)

temp = pd.DataFrame(df.groupby(['Risk Category bin'], axis=0, as_index=False)['Net_Loss'].sum())
plt.figure(figsize=(8,4))
sns.barplot(x="Risk Category bin", y="Net_Loss",data=temp)
plt.xticks(rotation=17.5)
plt.tight_layout()

temp = df.pivot_table(values=['Net_Loss'], index=['Year'], columns=['Region'], aggfunc='sum')
temp.plot(figsize=(8,4))

temp = df.pivot_table(values=['Net_Loss'], index=['Year'], columns=['Business bin'], aggfunc='sum')
temp.plot(figsize=(8,4))

temp = df.pivot_table(values=['Net_Loss'], index=['Year'], columns=['Risk Category bin'], aggfunc='sum')
temp.plot(figsize=(8,4))

df['x'] = df['Region'] + '---' + df['Business bin'] + '---' + df['Risk Category bin']
temp = pd.DataFrame(df.groupby(['x'], axis=0, as_index=False)['Net_Loss'].sum())
temp.sort_values('Net_Loss',axis=0, ascending=False, inplace=True)

plt.figure(figsize=(8,4))
sns.barplot(x="x", y="Net_Loss",data=temp.head())
plt.xticks(rotation=17.5)
plt.tight_layout()



