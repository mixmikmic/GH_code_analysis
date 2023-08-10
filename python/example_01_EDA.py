import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()

df =  pd.DataFrame( data=boston.data )
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df['MEDV'] = boston.target

df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Show correlation between each other
sns.set(style='whitegrid', context='notebook', font_scale=2)
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV' ]
sns.pairplot(df[cols], size=5)
plt.show()

# Generate single histogram & scatter plots
# Note : close() figure always, since seaborn will consume memory
plt.ioff() # turn off show()
for idx1, var1 in enumerate(list(df)):
    print var1+'....'
    plt.figure(figsize=(10, 8))
    sns.distplot(df[var1], kde=False, rug=False);
    plt.savefig('./figs/hist_%s.%s'%( var1,'png') )
    plt.close() 
    
    for idx2 in range(idx1+1, len(df.columns)):
        var2 = df.columns[idx2]
        corr = np.corrcoef(df[var1], df[var2])[0,1]
        sns.jointplot(x=var1, y=var2, data=df, size=10, color='g' if abs(corr) < 0.6 else 'r')
        plt.savefig('./figs/2D_Scatter_%s_%s.%s'%( var1, var2, 'png') )
        plt.close()
        

plt.ion() # Re-turn on the show()

cm = np.corrcoef(df.values.T)
#print cm

plt.figure(figsize=(15, 13))
labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
sns.set(font_scale=1.5)
sns.heatmap( cm, 
             cbar=True, 
             annot=True, 
             square=True, 
             fmt='.2f', 
             annot_kws={'size':15}, 
             yticklabels=labels, 
             xticklabels=labels)
plt.savefig('figs/correlation.png')

# LSTAT has larger correlation but looks not 1-order linear
plt.scatter(df[['LSTAT']], df['MEDV'])
plt.show()

# Some feature can be transfer with mathmatical way 
X_sqrt = np.sqrt(df[['LSTAT']])
y_log = np.log(df['MEDV'])
plt.scatter( X_sqrt, y_log ) 
plt.show()

cm = np.corrcoef( X_sqrt.T, y_log )
print cm

