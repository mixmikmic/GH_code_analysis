# import the library
import pandas as pd

# I usually define the working directory here, 
# so I don't have to type this long name over and over again
# Alternatively, you can redefine your working directory using os.chdir()
path0 = '/users/dten0001/Google Drive/data_archive/FluB/'

d0 = pd.read_csv(path0+"flub_500.csv")

# How many rows and columns does your dataframe have?
print(d0.shape)

# What are the column names?
print("Column names:")
print(d0.columns)

# Preview the first 10 rows of data. 
# If you leave the input parameter empty, default is 5
d0.head(10)

# Read one column, say, 'HA', into a list:
HA_col = list(d0["HA"])

# Or select multiple columns from d0 using a list of column names as input:
d_temp = d0[["iso_name", "HA", "MP", "NA"]]
d_temp.head()

# Select only the records from GISAID, and assign it to another dataframe, called d1
d1 = d0.loc[d0["data_source"] == "GISAID"]

# Select only records from Australia, New Zealand, and Singapore
countries = ["Australia", "New Zealand", "Singapore"]
d1 = d0.loc[d0["country"].isin(countries)]

# Select by multiple conditions: say, records from Australia, NZ and SG, from 2012 to 2014
# Currently, all data in d0 are strings. 
# We want to convert the collection year column, cyear, to a number (integer):
pd.to_numeric(d0["cyear"])

d1 = d0.loc[(d0["country"].isin(countries)) & (d0["cyear"] <= 2014) & (d0["cyear"] >= 2012)]
d1.head()

# How to see the sizes of different partitions of data, say, by collection year?
d0.groupby(["cyear"]).size()

# For multiple levels of grouping, say, by continent, then country:
d1 = d0.groupby(["continent", "country"]).size()
# Set to a dataframe
d1 = d1.reset_index()
# Give it some nice column names
d1.columns=["continent", "country", "counts"]

# preview it
d1.head(15)

# Note that if d1 is too big, only the top and bottom bits will be shown in Jupyter. 
# To get around this, increase the maximum number of rows printed out to, say, 500:
pd.set_option('display.max_rows', 500)
# Other options of this sort:
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# How to subsample? Select 200 records at random, without replacement:
d_sample = d0.sample(n=200, replace=False)

# Or select a percentage, like 20%:
d_sample = d0.sample(frac=0.2, replace=False)

# Write d_sample to a csv:
d_sample.to_csv(path0+"d_sample.csv", index=False)

