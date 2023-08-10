# import pandas with the nickname pd
import pandas as pd
# import numpy with the nickname np. 
import numpy as np
# import matplotlib with the the pyplot toolbox with the nickname plt. 
import matplotlib.pyplot as plt
# make plots appear inline
get_ipython().magic('matplotlib inline')

# read in your data 
df = pd.read_csv('sample_mods_scores.csv')

# see the top rows of your dataframe
df.head()

# see the last rows of your DataFrame
df.tail()

# return specific rows using numeric indices
df.iloc[3713:3716]

df

# get summary information about your DataFrame
df.info()
select = df

# check out basic statistics
df.describe()

# total scores grouped by division
totalscores_by_division = df.groupby('division')
totalscores_by_division['total'].agg([len, np.mean])

# total scores within a specific division in the group
totalscores_by_division.get_group('General Research Division').head()

# check identifier scores
df.identifier.value_counts()

# return the number of records missing a date element
print 'how many records are missing date element?'
nodates = df[(df.date == 0.0)]
print len(nodates)

# return the number of records missing a date element that have an identifier element
print 'how many records missing a date element have an identifier element?'
nodates_with_ids = df[(df.date == 0.0) & (df.identifier == 1.0)]
print len(nodates_with_ids)

# get all missing dates with collection and uuid info
df[df['date'] == 0.0][df['identifier'] == 1.0][['uuid','collection']].head()

# this table pivots identifier scores by division
dates_table = pd.pivot_table(df,index=['division'],columns=['date'], aggfunc={'date':len},fill_value=0)
dates_table

ids_by_coll = pd.pivot_table(df,index=['collection'],columns=['identifier'],aggfunc={'identifier':len},fill_value=0)
ids_by_coll

# export your data as a csv file
ids_by_coll.to_csv('ids_by_coll.csv')

# make a pie chart of identifier value counts
ids_table = df.identifier.value_counts()
ids_table.plot(kind='pie', figsize=(5, 5))

# create a stacked bar chart plotting identifier score counts by division
ids_table = pd.pivot_table(df,index=['division'],columns=['identifier'], aggfunc={'identifier':len},fill_value=0)
idscoresplot = ids_table.plot(kind='bar', stacked=True)

# create a bar chart viewing mean date scores by division
dates_by_division = df[['date', 'division']].groupby('division')
mean_dates_by_division = dates_by_division.mean()
dates_plot = mean_dates_by_division.sort(columns='date', ascending=False).plot(kind='bar', figsize=(10,5))
dates_plot.set_xlabel('Division')
dates_plot.set_ylabel('% of Items with a Date element')





















