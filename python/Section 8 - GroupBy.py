import pandas as pd

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
fortune.head()

# Group by Sector - returns a GroupBy object
sectors = fortune.groupby("Sector")
sectors

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
sectors = fortune.groupby("Sector")

# How many groups do we have?
len(sectors)

fortune['Sector'].nunique()

# Return a series showing how many rows are in each group
# `.size()` is very similar to .value_counts() on a series but is not sorted like .value_counts(), instead it is 
# sorted alphabetically
sectors.size()

fortune["Sector"].value_counts()

# Get first row of each group
sectors.first()

# Get last row of each group
sectors.last()

# Show a list of all groups (returns a dictionary)
# keys are each group, values are the indexes of rows in that group
sectors.groups

# First item in Aerospace and Defense
fortune.loc[24]

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
sectors = fortune.groupby("Sector")

# Extract a specific group
sectors.get_group("Energy")

# Non-group by equivilant
fortune[fortune["Sector"] == "Energy"]

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
sectors = fortune.groupby("Sector")

# max - Looks as left most value of each dataframe in groupby object and returns that row it (Company in this case)
#   Since this is a string, it will return the row that has the higest alphabetical ranking (closest to end of alphabet)
sectors.max()

sectors.min()

# sum - only applies to numbers, not strings
sectors.sum()

sectors.mean()

# Perform one of these on a specific column
sectors["Revenue"].sum()

sectors["Employees"].sum()

sectors["Profits"].max()

# Can perform across multiple columns
sectors[["Revenue", "Profits"]].sum()

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
# Can group by multiple columns
sectors = fortune.groupby(["Sector", "Industry"])
fortune.head()

# Now we have a multiIndex series
sectors.size()

sectors.sum()

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
sectors = fortune.groupby("Sector")

# Expects a dictionary 
#  Key is column name, value is method to perform
sectors.agg({"Revenue": 'sum', 
             "Profits": "sum", 
             "Employees": "mean"})

# Provide a list of values to the agg method
# Provides multiple methods to multiple columns
sectors.agg(["size", "sum", "mean"])

# Combine above two examples
sectors.agg({"Revenue": ["sum", "mean"], 
             "Profits": "sum", 
             "Employees": "mean"})

fortune = pd.read_csv('datasets/fortune1000.csv', index_col = "Rank")
sectors = fortune.groupby("Sector")

# Extract companies with highest profit per sector
sectors["Profits"].max()

# Create a new DF with same columns as fortune DF
df = pd.DataFrame(columns = fortune.columns)
# Creating an empty dataframe
df

# loop over groupbyobject requires two variables - first is for group and second is for dataframe in group
for sector, data in sectors:
    highest_revenue_company_in_group = data.nlargest(1, "Revenue")
    # this isn't an inplace operation
    df = df.append(highest_revenue_company_in_group)

# Show our companies per sector with highest revenue
df

# Now want to find highest revenue per city
cities = fortune.groupby("Location")
df1 = pd.DataFrame(columns = fortune.columns)

for city, data in cities:
    highest_revenue_in_city = data.nlargest(1, "Revenue")
    df1 = df1.append(highest_revenue_in_city)
df1

