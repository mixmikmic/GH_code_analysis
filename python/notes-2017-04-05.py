import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

tax = pd.read_csv("FEDS.csv",encoding='latin',usecols=[0,1,2,3,4,5,6])

tax.head()

tax.info()

tax['Prov/Terr'].unique()

tax = tax[tax['Prov/Terr'] != 'TOTAL']

tax['Prov/Terr'].unique()

tax.groupby('Prov/Terr')['Total Income'].sum().sort_values().plot(kind='bar')

tax.plot(kind='scatter',x='Total Income',y='Total')

tax[tax['Total Income'] > 1e10]

tax['Province'] = tax['Prov/Terr'].map(lambda code : int(code[0]))

tax.plot(kind='scatter',x='Total Income',y='Total',c='Province',figsize=(15,10),colormap='Accent',s=100,alpha=0.7)

