import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
1,2,3,4
5,6,,8
9,,,12
13,14,15,16'''

df = pd.read_csv(StringIO(unicode(csv_data)))
print df

df.isnull().sum()

# drop rows where atleast one column is NaN
df.dropna()

# drop rows that have all columns NaN
df.dropna(how='all')

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

# drop columns that have at least one NaN in any row by setting the axis argument to 1
df.dropna(axis=1)

from sklearn.preprocessing import Imputer

#Mean IMR
mean_imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
mean_imr = mean_imr.fit(df)
mean_imputed_df = mean_imr.transform(df.values)
print "Mean imputed DF:\n", mean_imputed_df

# MEDIAN IMR
median_imr = Imputer(missing_values='NaN', strategy='median', axis=0)
median_imr = median_imr.fit(df)
median_imputed_df = median_imr.transform(df.values)
print "Median imputed DF:\n", median_imputed_df

# MOST_FREQUENT IMR
most_frequent_imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
most_frequent_imr = most_frequent_imr.fit(df)
most_frequent_imputed_df = most_frequent_imr.transform(df.values)
print "Most Frequent imputed DF:\n", most_frequent_imputed_df

