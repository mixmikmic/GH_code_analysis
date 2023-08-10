import pandas as pd
import numpy as np
# This is to print in markdown style
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

# Read dataset locally:
#df = pd.read_csv('../data/BG-db.csv',index_col=0)


# Read dataset from url:
import io
import requests
url="https://raw.githubusercontent.com/trangel/stats-with-python/master/data/BG-db.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')),index_col=0)


del df['BG 3']
df.head()

# Let's visualize the data
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="ticks")


bins = np.arange(70,150,6)

A = df['BG 1'].values
B = df['BG 2'].values

# Show the results of a linear regression within each dataset
ax1 = sns.distplot(A,bins=bins,label='Before treatment')
ax2 = sns.distplot(B,bins=bins,label='After treatment')

plt.pyplot.xlabel('BG level')
plt.pyplot.ylabel('Distribution of BG levels')
plt.pyplot.legend(bbox_to_anchor=(0.45, 0.95), loc=2, borderaxespad=0.)

plt.pyplot.xlim((60,160))
plt.pyplot.show()

def difference(a,b):
    return b-a
df['Difference']=df.apply(lambda row: difference(row['BG 1'], row['BG 2']), axis=1)
df.head()

diff = df['Difference'].values

d = diff.mean()
std = diff.std(ddof=1)

printmd('$\overline{{d}} = {}$'.format(round(d,3)))
printmd('$s_d = {}$'.format(round(std,3)))

SE = std/(len(df))**0.5
DF = len(df)-1

# Mean of sample 1:
mu1 = df['BG 1'].values.mean()
mu2 = df['BG 2'].values.mean()

printmd('SE = {}'.format(round(SE,3)))
printmd('$\mu_1 = {}, \mu_2 = {}$'.format(round(mu1,2),round(mu2,2)))

TestStatistic = (mu2 - mu1)/SE

from scipy import stats
pvalue = 2.0* stats.t.cdf(TestStatistic, DF)
# Multiply by two, since this is two-tailed test

printmd('t-score {}'.format(round(TestStatistic,2)))
printmd("p-value = {}".format(round(pvalue,5)))

# Sample 1
a = df['BG 2']
b = df['BG 1']

stats.ttest_rel(a,b)

