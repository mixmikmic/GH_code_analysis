import seaborn as sns
import numpy as np

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

tips = sns.load_dataset('tips')

tips.head()

lmPlot= sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','x'])

lmPlot= sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',markers=['o','x'],scatter_kws={'s':100})

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex',row='time')



sns.lmplot(x='total_bill',y='tip',data=tips,col='day',row='time',hue='sex')

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex')

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',aspect=0.6,size=8)





sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',aspect=0.6,size=8)



