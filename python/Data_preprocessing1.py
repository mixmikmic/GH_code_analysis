from pandas import *
import pandas as pd
a = pd.Series (['foo', 'foo', 'foo', 'foo', 'bar', 'bar','bar', 'bar', 'foo', 'foo', 'foo'], dtype=object)
b = pd.Series (['one', 'one', 'one', 'two', 'one', 'one','one', 'two', 'two', 'two', 'one'], dtype=object)
c = pd.Series (['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny','shiny', 'dull', 'shiny', 'shiny', 'shiny'], dtype=object)
crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])

### Descriptive data summarization
import scipy
from scipy.stats import *
x=[1,5,7,8,9,2,3,5,7,3,5,9]
y=[2,4,5,1,3,3,7,8,7,6,4,5]
print(describe(x))

print(describe(y))

z = DataFrame([[1.4, np.nan], [7.1, -4.5],[np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
print(z.describe())

### Row/column wise statistics
print("Row Mean=")
print(z.mean(axis=1)) ### Row wise
print ("Column Mean=")
print (z.mean(axis=0)) ### Column wise

### Histogram, box plot and scatter plot
from scipy import stats
from pylab import *
import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax1.hist(x,bins=5,color='green',alpha=0.3)
ax1.set_xlim([0,10])
ax1.set_ylim([0,5])
ax1.set_title('Histogram')
ax2.boxplot(x,0,'gD')
ax2.set_title('Box plot')
ax2.set_ylim([-1,10])
ax3.scatter(x,y)
ax3.set_title('Scatter plot')
show()

### Probability Mass Function
import pandas as pd
x=[1,5,7,8,9,2,3,5,7,3,5,9]
df = pd.DataFrame([1,5,7,8,9,2,3,5,7,3,5,9],columns=['Probability Mass Function'])
df.plot(kind='density',title='Density plot')
show()

### Q-Q Plot for two samples
import statsmodels.api as sm
import numpy as np
x=np.array([1,5,7,8,9,2,3,5,7,3,5,9])
y=np.array([2,4,5,1,3,3,7,8,7,6,4,5])
pp_x = sm.ProbPlot(x,fit=True)
pp_y = sm.ProbPlot(y,fit=True)
fig=pp_x.qqplot(line='45', other=pp_y)
plt.show()

### Side-by-Side Boxplot
df = pd.DataFrame(np.random.rand(10,2), columns=['Col1', 'Col2'] )
df['X'] = pd.Series(['A','A','A','A','A','B','B','B','B','B'])
fig2=plt.figure()
bp = df.boxplot(by='X')
show()

