import pandas as pd

url = 'http://bit.ly/drinksbycountry'
drinks = pd.read_csv(url)

drinks.head()

drinks.dtypes

# to change use .astype() 
drinks['beer_servings'] = drinks.beer_servings.astype(float)

drinks.dtypes

drinks = pd.read_csv(url, dtype={'beer_servings':float})

drinks.dtypes

url = 'http://bit.ly/chiporders'
orders = pd.read_table(url)

orders.head()

orders.dtypes

# we use .str to replace and then convert to float
orders['item_price'] = orders.item_price.str.replace('$', '').astype(float)

orders.dtypes

# we can now calculate the mean
orders.item_price.mean()

orders['item_name'].str.contains('Chicken').head()

# convert to binary value
orders['item_name'].str.contains('Chicken').astype(int).head()

