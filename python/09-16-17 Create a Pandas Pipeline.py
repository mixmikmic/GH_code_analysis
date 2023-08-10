import pandas as pd

# Create empty dataframe
df = pd.DataFrame()

# Create a column
df['name'] = ['John', 'Steve', 'Sarah']
df['gender'] = ['Male', 'Male', 'Female']
df['age'] = [31, 32, 19]

# View dataframe
df

# Create a function that
def mean_age_by_group(dataframe, col):
    # groups the data by a column and returns the mean age per group
    return dataframe.groupby(col).mean()

print (mean_age_by_group(df,'gender'))

# Create a function that
def uppercase_column_name(dataframe):
    # Capitalizes all the column headers
    dataframe.columns = dataframe.columns.str.upper()
    # And returns them
    return dataframe

print (uppercase_column_name(df))

# Create a pipeline that applies the mean_age_by_group function
(df.pipe(mean_age_by_group, col='gender')
   # then applies the uppercase column name function
   .pipe(uppercase_column_name)
)



