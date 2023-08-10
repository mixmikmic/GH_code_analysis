import pandas as pd

# read a dataset of movie reviewers (modifying the default parameter values for read_table)
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
url = 'http://bit.ly/movieusers'
users = pd.read_table(url, sep='|', header=None, names=user_cols, index_col='user_id')

users.head()

users.shape

# use df.cat_name.duplicated()
# output True if row above is the same
users.zip_code.duplicated()

# type
type(users.zip_code.duplicated())

# we can use .count() since it's a series
# there're 148 duplicates
users.zip_code.duplicated().sum()

# it will output True if entire row is duplicated (row above)
users.duplicated()

# examine duplicated rows
users.loc[users.duplicated(), :]

# keep='first'
# mark duplicates as True except for the first occurence
users.loc[users.duplicated(keep='first'), :]

# keep='last'
# 7 rows that are counted as duplicates, keeping the later one

# this is useful for splitting the data
users.loc[users.duplicated(keep='last'), :]

# mark all duplicates as True
# this combines the two tables above
users.loc[users.duplicated(keep=False), :]

# drops the 7 rows
users.drop_duplicates(keep='first').shape

# drops the last version of the 7 duplicate rows
users.drop_duplicates(keep='last').shape

# drops all 14 rows
users.drop_duplicates(keep=False).shape

# only consider "age" and "zip_code" as the relevant columns
users.duplicated(subset=['age', 'zip_code']).sum()

