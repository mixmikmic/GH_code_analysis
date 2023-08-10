import re
import pandas as pd

# Create a dataframe with a single column of strings
data = {'raw': ['Arizona 1 2014-12-23       3242.0',
                'Iowa 1 2010-02-23       3453.7',
                'Oregon 0 2014-06-20       2123.0',
                'Maryland 0 2014-03-14       1123.6',
                'Florida 1 2013-01-15       2134.0',
                'Georgia 0 2012-07-14       2345.6']}
df = pd.DataFrame(data, columns = ['raw'])
df

# Which rows of df['raw'] contain 'xxxx-xx-xx'?
df['raw'].str.contains('....-..-..', regex=True)

# In the column 'raw' extract single digit in the strings
df['female'] = df['raw'].str.extract('(\d)', expand=True)
df['female']

# In the oclumn 'raw', extract xxxx-xx-xx in the strings
df['date'] = df['raw'].str.extract('(....-..-..)', expand=True)
df['date']

# in the column 'raw' extract ####.## in the strings
df['score'] = df['raw'].str.extract('(\d\d\d\d\.\d)', expand=True)
df['score']

# In the column 'raw', extract the word in the strings
df['state'] = df['raw'].str.extract('([A-Z]\w{0,})', expand=True)
df['state']

df

