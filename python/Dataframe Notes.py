import pandas as pd
import numpy as np

# read-in the data frame from csv file
DF=pd.read_csv('../DATA/backtesting.csv')

# returns a tuple with number of rows/columns
DF.shape

DF.head(3) #only the 3 first lines are shown

DF.columns

RSI=DF[['RSI']]

a=DF[['RSI','Ranging']]

DF.iloc[1, 0]

reversed_true=DF.loc[DF['Reversed']==True]

DF.loc[(DF['Reversed']==True) | DF['Divergence']==True]

div_class=pd.crosstab(DF['Divergence'], DF['Reversed'],margins=True)

print(div_class)

div_class/div_class.loc["All"]

DF.groupby(['Divergence']).agg({'Trend length after (bars)': 'mean'})

pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3, retbins=True)

cuts = np.array([0,2,4,6,10])
pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), cuts)

