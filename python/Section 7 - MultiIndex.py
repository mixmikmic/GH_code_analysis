import pandas as pd

# This module uses the Big Mac Index dataset
bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'])
bigmac.head()

bigmac.info()

bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'])

# Can pass multiple values to `set_index()`
bigmac.set_index(keys=["Date", "Country"])

# Can change how the indexes are structured by changing order of passed index columns
bigmac.set_index(keys = ['Country', 'Date'])

bigmac.set_index(keys=["Date", "Country"], inplace=True)
bigmac.head()

# Sorting the index, will sort all indexes
bigmac.sort_index(inplace=True)
bigmac.head()

# Index attribute will return a `MultiIndex` object representing each level of the index
bigmac.index

# Get the index names
bigmac.index.names

# MultiIndexes are a different object type from single index dataframes 
type(bigmac.index)

# Indexes are a tuple of all levels of the index
# Because of this, we'll need to pass a tuple to get specific cell values in the next steps
bigmac.index[0]

# Set index while importing and sort to improve speed
bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'], index_col = ['Date', 'Country'])
bigmac.sort_index(inplace=True)
bigmac.head()

# get_level_values() is called on the index
# Accepts either a numeric value or a name
# This function does not de-duplicate a layer
bigmac.index.get_level_values('Date')

# Pass a numeric value (0 = Date, 1 = Country)
bigmac.index.get_level_values(1)

bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'], index_col = ['Date', 'Country'])
bigmac.sort_index(inplace=True)
bigmac.head()

# Change "Date" to "Day" and "Country" to "Location"
bigmac.index.set_names(['Day', 'Location'], inplace=True)
bigmac.head()

bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'], index_col = ['Date', 'Country'])

# By default, it will sort all indexes in ascending order
bigmac.sort_index()

# Alter sort to date in ascending order by county in descending by passing a list to `ascending` parameter
bigmac.sort_index(ascending = [True, False], inplace=True)
bigmac.head()

bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'], index_col = ['Date', 'Country'])
bigmac.sort_index(inplace=True)

# Pass a tuple of the combination of indexes we are looking for
# If we put 1 layer, it pulls everything for that index
bigmac.loc[('2010-01-01')]

# Extract something specific - Date/Country combination needed
bigmac.loc[('2010-01-01', 'Brazil')]

# Extract something specific - Date/Country combination needed - and pull only 1 column instead of entire series
bigmac.loc[('2010-01-01', 'Brazil'), "Price in US Dollars"]

# Extract 2015-07-01
bigmac.loc['2015-07-01']

# Using the `ix` method, doesn't show outer layer - Notice "Date" is gone
bigmac.ix[('2016-01-01')]

# Return a series
bigmac.ix[('2016-01-01', 'Brazil')]

# Get specific value instead of entire series
bigmac.ix[('2016-01-01', 'Brazil'), "Price in US Dollars"]

bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'], index_col = ['Date', 'Country'])
bigmac.sort_index(inplace=True)

# Doesn't have an inplace, so it needs to be reassigned
bigmac = bigmac.transpose()
bigmac.head()

# Extract 
#  Now the index is only single level - Price in US Dollars
# This will return a multiple index series
bigmac.ix["Price in US Dollars"]

# Extract specific value by passing an index to the second parameter
bigmac.ix['Price in US Dollars', ('2010-01-01','Brazil')]

# Return a series for date
bigmac.ix['Price in US Dollars', ('2010-01-01')]

bigmac = pd.read_csv('datasets/bigmac.csv', parse_dates=['Date'], index_col = ['Date', 'Country'])
bigmac.sort_index(inplace=True)
bigmac.head()

# No inplace parameter, so it needs to be reassigned
bigmac = bigmac.swaplevel()
bigmac.head()

world = pd.read_csv('datasets/worldstats.csv', index_col = ["country", "year"])
world.head()

world.stack()

# Convert the above series to a dataframe instead of to a series
# Notice we have a column with no name so it was given "0"
world.stack().to_frame()

world = pd.read_csv('datasets/worldstats.csv', index_col = ["country", "year"])
s = world.stack()
s.head()

# Undo the stack
s.unstack()

# Can keep unstacking
#  This will create a multiindex column
#  In this case having a Population with year subgroups and GDP with year subgroups
s.unstack().unstack()

# Can unstack once more
s.unstack().unstack().unstack()

world = pd.read_csv('datasets/worldstats.csv', index_col = ["country", "year"])

s = world.stack()
s.head()

# unstack
#  Provide an index (name or numeric index) that we want to unstack
#   Country =0; Year =1; Unnamed (Pop/GDP) = 2
s.unstack("year")

# Providing negative arguments will start at inner most level and work from there
#  This will do the same as above (because year is the second from the inner most)
s.unstack(-2)

world = pd.read_csv('datasets/worldstats.csv', index_col = ["country", "year"])
s = world.stack()
s.head()

# Can provide a list of levels to unstack
#  Pull out year then country to columns
s.unstack(level = [1, 0])

# Order is important
s.unstack(level=[0,1])

# fill_value parameter - Fill in NaN with specific value
#   Notice the Albania values in the 1960s below
s = s.unstack("year", fill_value=0)
s.head()

sales = pd.read_csv('datasets/salesmen.csv', parse_dates = ['Date'])
sales["Salesman"].astype('category')
sales.head()

# pivot so that the salesman is a column and renenue is part of that column - Creates a new dataframe
#   index = What to use as the index (unchanged in this case)
#   columns = What columns will be used as the new columns; The more unique values here, the more columns
#   values = What are going to be the values at the index/column intersection
sales.pivot(index="Date", columns="Salesman", values="Revenue")

foods = pd.read_csv('datasets/foods.csv')
foods.head()

# Parameters
#  values = Column we are performing aggfunc on
#  index = Group by...
#  aggfunc = (Default is 'mean')
foods.pivot_table(values="Spend", index="Gender", aggfunc='mean')

# Output shows breakdown of average spend by Gender

# Total spend by gender
foods.pivot_table(values="Spend", index="Gender", aggfunc='sum')

# Total sales by item
foods.pivot_table(values="Spend", index="Item", aggfunc='sum')

# Can provide a list to `index` to get further break down
#  Total sales by gender per item
foods.pivot_table(values="Spend", index=["Gender", "Item"], aggfunc='sum')

# columns parameter
# Want to look at it by Gender/Item and City and create a dataframe based on unique column values in the columns field
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='sum')

# Can have a multiindex column too
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = ["Frequency","City"], aggfunc='sum')

# Other aggregations
#   sum
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='sum')

# mean
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='mean')

# count
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='count')

# max
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='max')

# min
foods.pivot_table(values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='min')

# pivot_table is available directly on the pandas object too (not just a dataframe)
# Requires you pass a dataframe as the first parameter
pd.pivot_table(data=foods, values="Spend", index=["Gender", "Item"], columns = "City", aggfunc='min')

sales = pd.read_csv('datasets/quarters.csv')
sales

# melt works on the pandas object
#   frame = Dataframe we are operating on
#   id_vars = Column that will be preserved - in this case, we want to keep the salesmen, the Q1-Q4 will be moved into table
# By default, the new columns will be "variable" and "value"
#   var_name = Name of new variable column
#   value_name = Name of new value column
pd.melt(frame=sales, id_vars="Salesman", var_name="Quarter", value_name="Revenue")



