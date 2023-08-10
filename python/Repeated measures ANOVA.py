import pandas as pd
import numpy as np
# This is to print in markdown style
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

# Read database locally
#df = pd.read_csv('../data/BG-db.csv',index_col=0)

# Read dataset from url:
import io
import requests
url="https://raw.githubusercontent.com/trangel/stats-with-python/master/data/BG-db.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')),index_col=0)


df.columns=['before','during','after']
df.index.name='Subject'
df.head(10)

# Let's visualize the data
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="ticks")


bins = np.arange(70,150,6)

A = df['before'].values
B = df['during'].values
C = df['after'].values


# Show the results of a linear regression within each dataset
ax1 = sns.distplot(A,bins=bins,label='Before treatment')
ax2 = sns.distplot(B,bins=bins,label='During treatment')
ax3 = sns.distplot(C,bins=bins,label='After treatment')


plt.pyplot.xlabel('BG level')
plt.pyplot.ylabel('Distribution of BG levels')
plt.pyplot.legend(bbox_to_anchor=(0.45, 0.95), loc=2, borderaxespad=0.)

plt.pyplot.xlim((60,160))
plt.pyplot.show()

# Calculate means for each group
mu1 = df['before'].values.mean()
mu2 = df['during'].values.mean()
mu3 = df['after'].values.mean()

#Grand mean
mu = df.values.mean()

printmd('$\overline{{x}}_1 = {}$'.format(round(mu1,3)))
printmd('$\overline{{x}}_2 = {}$'.format(round(mu2,3)))
printmd('$\overline{{x}}_3 = {}$'.format(round(mu3,3)))
printmd('$\overline{{x}} = {}$'.format(round(mu,3)))

# Number of samples for each group:
n = len(df)
# Here the number of samples is the same for all groups:
n1 = n
n2 = n
n3 = n

# SS groups:
SSgroups= n1*(mu1-mu)**2 + n2*(mu2-mu)**2 + n3*(mu3-mu)**2
printmd("$SS_{{groups}} = {}$".format(round(SSgroups,3)))

# Group 1 is a column of the dataset (Before):   
group1 = df['before'].values
# Take mean value for this group
mu1 = group1.mean() 
# Now calculate sum (x - mu1)^2 for all x values in this group:
ss_group1 = np.sum((group1-mu1)**2)
printmd('SS for group 1 ={}'.format(ss_group1))

group2 = df['during'].values
# Take mean value for this group
mu2 = group2.mean() 
# Now calculate sum (x - mu1)^2 for all x values in this group:
ss_group2 = np.sum((group2-mu2)**2)
printmd('SS for group 2 ={}'.format(ss_group2))
#
group3 = df['after'].values
# Take mean value for this group
mu3 = group3.mean() 
# Now calculate sum (x - mu1)^2 for all x values in this group:
ss_group3 = np.sum((group3-mu3)**2)
printmd('SS for group 3 ={}'.format(ss_group3))
#
SSw = ss_group1 + ss_group2 + ss_group3
printmd('$SS_w = {}$'.format(round(SSw,3)))

# Let's calculate the subject means, i.e., the means for each row in our dataframe  

def subject_mean(a,b,c):
    '''Returns mean value of 3 numbers'''
    return float((a+b+c)/3.0)

df['Subject means'] = df.apply(lambda row: subject_mean(row['before'],row['during'],row['after']), axis = 1)
df.head()

k=3.0
subject_means = df['Subject means'].values
SSsubjects = k * np.sum( (subject_means - mu)**2 )

printmd('$SS_{{subjects}} = {}$'.format(round(SSsubjects,3)))

SSerror = SSw - SSsubjects
printmd('$SS_{{error}} = {}$'.format(round(SSerror,3)))

# Degrees of freedom for groups:
df1 = k-1
# Degrees of freedom for error
df2 = (n-1)*(k-1)

MSgroups = SSgroups/df1

MSerror = SSerror/df2

Fstatistic = MSgroups/MSerror

printmd('$MS_{{groups}} = {}$'.format(round(MSgroups,3)))
printmd('$MS_{{error}} = {}$'.format(round(MSerror,3)))
printmd('$F$-statistic $= {}$'.format(round(Fstatistic,3)))
printmd('DF1 = {}, DF2 = {}'.format(int(df1),int(df2)))



