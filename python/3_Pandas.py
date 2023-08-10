# Import statement
import pandas as pd

sales = [('Jones LLC', 150, 200, 50),
         ('Alpha Co', 200, 210, 90),
         ('Blue Inc', 140, 215, 95)]
labels = ['account', 'Jan', 'Feb', 'Mar']
df = pd.DataFrame.from_records(sales, columns=labels)

print (df)

# Import statement
import pandas as pd

# Dictionary with data of the BRICS nations
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

brics = pd.DataFrame(dict)
print(brics)

# Set the index for brics
brics.index = ["BR", "RU", "IN", "CH", "SA"]

# Print out brics with new index values
print(brics)

df = pd.read_json('https://api.github.com/repos/pydata/pandas/issues?per_page=5')
print (df.head())

# Read file with delimiter specified (default delimited = ',')
# Download at http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
wine_data = pd.read_csv('wine.csv', sep=';')

# View the first few lines
wine_data.head()

# View the last 3 lines
wine_data.tail(3)

# View the dimensions
wine_data.shape

# Select rows from begin index to end index
some_data = wine_data[10:30]
print (some_data)

# First five entries. Show only columns mentioned
wine_data.loc[:5,["alcohol", "quality"]] # Provide list of columns

# Index a particular column using its name
wine_data['quality']

# Column wise mean
wine_data.mean()

# Mean for a particular column
wine_data['pH'].mean()

# Find correlation between different columns
wine_data.corr()

# Boolean filter based on a column's value
good_quality = wine_data['quality'] > 6
print (good_quality)

# Use filter above to fetch the required rows
good_wines = wine_data[good_quality]
print (good_wines.shape)
print (good_wines)
x = [i for i in range(len(good_quality)) if good_quality[i]]
print (x)
print (wine_data[wine_data['quality'] > 6])

# Sort by a column 
good_wines_sorted = good_wines.sort_values(by='quality')#, ascending=False)
good_wines_sorted.head()

# Iterating over data frame elements
for index, row in good_wines_sorted.iterrows() :
    print(row['quality'], row['alcohol'])

