import pandas as pd
import numpy as np

df1 = pd.read_csv('titanic passenger list.csv') 
# Reading the dataset into a dataframe using Pandas
df1.head()

# We can boxplot an entire DataFrame, quick & nasty, ignores non-numeric data too
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

bp = df1.boxplot()

bp = df1.boxplot(column='fare')
# Sometimes you have to force display of outliers:
# bp = plt.boxplot(df1.fare, 0, 'ro') # red circle

# We can see a bunch of fares above 200, then something around 500, look at the outliers:
df1[df1['fare'] > 400] 

# fare of 512.3292 looks very high compared to other, that's a weird value too, did they split the cost?
# or was it just the one ticket?
512.3292 * 4

# so how do you get 512.3292 from £512, 6s
# 6shillings is ~1/3 pound?
# 6/20 is 0.3 so that may explain the 0.3292..
def poundit(p,s,d):
    return p + (d / 12.0 + s) / 20
# force the 12.0 not the 20!
# There were 20 shillings per pound and the shilling was divided into 12 pennies.

poundit(512,6,0) # £512, 6s

# plot by class
bp = df1.boxplot(column='fare', by = 'pclass')



