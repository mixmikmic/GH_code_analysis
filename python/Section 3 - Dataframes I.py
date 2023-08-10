import pandas as pd

nba = pd.read_csv('datasets/nba.csv')

# NBA Dataframe we'll use in this section
# Notes: 
#   - Has NaN values (for blanks in the CSVs)
#   - Has blank row at the end
#   - Index is generated automatically because we didn't specify
#   - Numeric values are shown as floats (not the ints as in the CSV)
#       This is done because there are NaN values in the column
nba

# head - Returns first rows similar to series
nba.head()

# tail - Returns last rows
nba.tail()

# index - Describes the index of the dataframe
nba.index

# values - Returns multidimensional array of all values in dataframe
nba.values

# shape - Python tuple of number of rows and colums
nba.shape

# dtypes - Describes each column (series object is returned)
nba.dtypes

# columns - Dataframe specific - Returns an array of column names
nba.columns

# axes - Dataframe specific - Returns index/columns results
nba.axes

# info - Summary of dataframe
nba.info()

# get_dtype_counts() - Returns `dtypes` portion from above
nba.get_dtype_counts()

rev = pd.read_csv('datasets/revenue.csv', index_col = "Date")
rev.head(3)

s = pd.Series([1,2,3])
s

s.sum()

# Sum on a dataframe results in a new Series, index = Columns and value = sum of everything in that column
rev.sum()

# Same result
rev.sum(axis = "index") # or rev.sum(axis = 0)

# To sum horizontally (in this case, by date instead of by city)
rev.sum(axis = "columns") # or rev.sum(axis = 1)

nba = pd.read_csv('datasets/nba.csv')

# "simpler" syntax
# Columns with singular names (no spaces)
# This is case sensitive
# use `.NAMEOFCOLUM`
# Returns a series

# This method only works when columns do not have spaces
nba.Name

nba.Number

# Second method (works when columns have spaces...and in any other case)
# Bracket syntax
nba["Name"]

# Can method chain on returned objects (series)
nba['Name'].head(3)

nba = pd.read_csv('datasets/nba.csv')
nba.head(3)

# Returns a new dataframe, not series
nba[["Name", "Team"]].head(3)

# Can change order of columns in returned frame
nba[["Team", "Name"]].head(3)

nba = pd.read_csv('datasets/nba.csv')
nba.head(3)

# If we assign a value to an unknown column, it will create a new column with that name and those values
# If the value DOES exist and we use equality, it will OVER WRITE existing values
# This always adds after existing columns

# Add a scalar value - Single value assigned to all
nba['Sport'] = "Basketball"
nba.head(3)

nba['League'] = "National Basketball Association"
nba.head(3)

# Reset for next example
nba = pd.read_csv('datasets/nba.csv')
nba.head(3)

# Use `insert` method
# loc - Index location within the columns (Name = 0, Team = 1, Number = 2, Position = 3, etc)

# Insert into 3rd location (between Number and Position)
nba.insert(3, column="Sport", value="Basketball")
nba.head(3)

# Insert between Height and Weight
nba.insert(7, column="League", value="National Basketball Association")
nba.head(3)

nba = pd.read_csv('datasets/nba.csv')
nba.head(3)

# add 5 to all ages
nba["Age"].add(5)

# Short hand for above
nba["Age"] + 5

# Subtract 5M from all in Salary
nba["Salary"] - 5000000  # or .sub()

# Convert weight from pounds to kilograms
nba["Weight"].mul(0.453592)

# Save new series to dataframe
nba['Weight in Kilograms'] = nba['Weight'] * 0.453592
nba.head(3)

# Salary in millions
nba['Salary in Millions'] = nba['Salary'].div(1000000)
nba.head(3)

nba = pd.read_csv('datasets/nba.csv')

nba['Team'].value_counts()

nba["Position"].value_counts()

nba = pd.read_csv('datasets/nba.csv')

nba.head(3)

nba.tail(3)

# dropna - Removes any rows from dataframe where any columns have NaN (by default)
#   determined by the `how` parameter; Default = any
nba.dropna()

# how = all
# Remove only where ALL columns are NaN
nba.dropna(how='all')

# Also uses `inplace`
nba.dropna(how='all', inplace=True)

# Drop COLUMN that has NaN
# Change `axis` value from default of 0/"index" to 1/"columns"
nba.dropna(axis=1)

# Only remove row is Null is in a specific column
# Use `subset` and provide a list of column names
nba.dropna(subset = ["Salary"])

nba = pd.read_csv('datasets/nba.csv')

# Default isn't always intuitive; This example replaces ALL NaN with 0 which doesn't make sense on "College" column
nba.fillna(0).head(10)

# Fill value on a specific column
nba["Salary"].fillna(0, inplace=True)
nba.head(10)

nba["College"].fillna("No College", inplace=True)
nba.head(10)

# Import nba and remove the last row which has all NaN
nba = pd.read_csv('datasets/nba.csv').dropna(how="all")

# First we need to deal with the NaN values
nba['Salary'].fillna(0, inplace=True)
nba['College'].fillna("None", inplace=True)
nba.head(6)

# In our original CSV, all of the numeric values are integers, but because of the NaN values they were imported as floats
# Show our types
nba.dtypes

# Show our types (alternative)
nba.info()

# Convert Salary to an int
# astype does not have an inplace parameter, so it needs to be reassigned to salary series
nba['Salary'] = nba['Salary'].astype("int")

nba.head(3)

nba['Age'] = nba['Age'].astype("int")
nba['Number'] = nba['Number'].astype("int")
nba['Weight'] = nba['Weight'].astype("int")
nba.head(3)

# Certain datatypes take of different amounts of space. Notice that we've reduced the memory footprint compared to above
nba.info()

# Category datatype
# Ideal when we have a small number of unique values in a column
#    In this example - Position column or Team column
nba['Position'].nunique()

nba['Position'] = nba['Position'].astype("category")
nba['Team'] = nba['Team'].astype("category")
nba.head(3)

# Check dtypes and notice we have two categories and we've reduced memory usage
nba.info()

nba = pd.read_csv('datasets/nba.csv')

# Sort entire DF only by the Name column
nba.sort_values("Name")

# By default NaN values are placed at the end 
# Can be changed from default of na_position = "last" to na_position = "first"
nba.sort_values("Salary")

nba = pd.read_csv('datasets/nba.csv')

# Sort by Team then Name in ascending order
nba.sort_values(["Team", "Name"])

# Reverse both sorts
nba.sort_values(["Team", "Name"], ascending = False)

# Sort team ascending, but names in descending
nba.sort_values(["Team", "Name"], ascending = [True, False])

nba = pd.read_csv('datasets/nba.csv')

# Mess up our dataframe so the indexes are out of place
nba.sort_values(["Number", "Salary", "Name"], inplace=True)
nba.tail(10)

# Sort the index in ascending order
nba.sort_index()

# Need to remove NaNs for rank to work
nba = pd.read_csv('datasets/nba.csv').dropna(how = "all")
nba['Salary'] = nba['Salary'].fillna(0).astype("int")
nba.head(3)

# Give the player with highest salary a rank of 1, second highest 2, etc
#   This default ranks in reverse order (highest salary is at the end) (fix by changing ascending)
nba['Salary Rank'] = nba['Salary'].rank(ascending=False).astype("int")
nba.head(5)

nba.sort_values(by="Salary", ascending=False)



