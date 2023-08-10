import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

flights.head()

tc= tips.corr()

sns.heatmap(tc)

sns.heatmap(tc,annot=True,cmap='coolwarm')

# Create a pivotal Table

fpivot =flights.pivot_table(index='month',columns='year',values='passengers',)
fpivot

### Generate a heat map of the table

sns.heatmap(fpivot,cmap='inferno')

sns.heatmap(fpivot,cmap='inferno_r')

sns.heatmap(fpivot,cmap='BuPu')

sns.heatmap(fpivot,cmap='inferno_r',linecolor='White',linewidths=2)

sns.clustermap(fpivot,cmap='inferno_r')

sns.clustermap(fpivot,cmap='coolwarm')

sns.clustermap(fpivot,cmap='coolwarm',standard_scale=1)







