import pandas as pd

data = pd.read_csv('data/fortune1000.csv', index_col='Rank')
data.head()

sectors = data.groupby('Sector') # creating a DataFrameGroupBy object

sectors.head()

data.head()

print type(data)
print type(sectors)

print len(data)
print len(sectors) # data frame for each sector

data['Sector'].nunique()

sectors.size() # occurrences of each sector
# data['Sector'].value_counts()

sectors.first() # first occurence of each grouping

sectors.last() # last occurence of each grouping

sectors.groups # Dict
# sectors.groups.keys() one key for each sector
# sectors.groups.values() List of Index 

data.loc[24]

sectors.get_group(name='Apparel') # return a dataframe

sectors.get_group(name='Energy') 

for g in sectors.groups.keys():
    print "Group: {}".format(g), sectors.get_group(g)

# Get Company in alphabetical order (first and last)
sectors.max() 
sectors.min()
sectors.sum() # only numbers --> sectors.get_group('Apparel')['Revenue'].sum()
sectors.mean()

# Series --> Sum for each sector of all the revenues
sectors['Revenue'].sum()

sectors['Revenue'].max() # max, min, mean

sectors[['Revenue', 'Employees']].sum()

sectors = data.groupby(['Sector', 'Industry'])

# Double Index
sectors.size()

sectors.sum() # --> MultiIndex DataFrame

sectors['Revenue'].sum()

sectors = data.groupby('Sector')

# On selected column apply function
sectors.agg({'Revenue': 'sum', 
             'Profits': 'sum',
             'Employees': 'mean'
            })

# For each columns apply all
sectors.agg(['size', 'sum', 'mean'])

# Mixing
# On selected column apply function
sectors.agg({'Revenue': ['sum', 'mean'],
             'Profits': 'sum',
             'Employees': 'mean'
            })

aggregated = sectors.agg({'Revenue': ['sum', 'mean'],
             'Profits': 'sum',
             'Employees': 'mean'
            })

aggregated.loc['Apparel']

aggregated['Revenue']['sum']

data = pd.read_csv('data/fortune1000.csv', index_col='Rank')
sectors = data.groupby('Sector')
data.head()

# new dataframe, used to store data
df = pd.DataFrame(columns=data.columns)
df

# highest revenue for each sector

for sector, value in sectors:
    highest_revenue_company_in_group = value.nlargest(1, 'Revenue')
    df = df.append(highest_revenue_company_in_group)
    

df

cities = data.groupby('Location')

df = pd.DataFrame(columns=data.columns)
df

for city, value in cities:
    highest_revenue_in_city = value.nlargest(1, 'Revenue')
    df = df.append(highest_revenue_in_city)

df





