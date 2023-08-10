import pandas as pd

url = 'http://bit.ly/drinksbycountry'
drinks = pd.read_csv(url)

drinks.head()

drinks.info()

# we can count the actual memory usage using the following command
drinks.info(memory_usage='deep')

# we can check how much space each column is actually taking
# the numbers are in bytes, not kilobytes
drinks.memory_usage(deep=True)

type(drinks.memory_usage(deep=True))

# since it is a series, we can use .sum()
drinks.memory_usage(deep=True).sum()

# there are only 6 unique values of continent
# we can replace strings with digits to save space
sorted(drinks.continent.unique())

drinks.continent.head()

# converting continent from object to category 
# it stores the strings as integers
drinks['continent'] = drinks.continent.astype('category')

drinks.dtypes

drinks.continent.head()

# .cat is similar to .str
# we can do more stuff after .cat
# we can see here how pandas represents the continents as integers
drinks.continent.cat.codes.head()

# before this conversion, it was over 12332 bytes
# now it is 584 bytes
drinks.memory_usage(deep=True)

# we can convert country to a category too
drinks.dtypes

drinks['country'] = drinks.country.astype('category')

# this is larger! 
# this is because we've too many categories
drinks.memory_usage(deep=True)

# now we've 193 digits
# it points to a lookup table with 193 strings!
drinks.country.cat.categories

# passing a dictionary {} to the DataFrame method = 
id_list =[100, 101, 102, 103]
quality_list = ['good', 'very good', 'good', 'excellent']
df = pd.DataFrame({'ID': id_list, 'quality': quality_list })
df

# this sorts using alphabetical order
# but there is a logical ordering to these categories, we need to tell pandas there is a logical ordering
df.sort_values('quality')

# how do we tell pandas there is a logical order?
quality_list_ordered = ['good', 'very good', 'excellent']
df['quality'] = df.quality.astype('category', categories=quality_list_ordered, ordered=True)

# here we have good < very good < excellent
df.quality

# now it sorts using the logical order we defined
df.sort_values('quality')

# we can now use boolean conditions with this
# here we want all columns where the row > good
df.loc[df.quality > 'good', :]

