import numpy as np
import pandas as pd

market_df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
market_df.head()

# Selecting the rows from indices 2 to 6
market_df[2:7]

# Selecting alternate rows starting from index = 5
market_df[5::2].head()

# Using df['column']
sales = market_df['Sales']
sales.head()

# Using df.column
sales = market_df.Sales
sales.head()

# Notice that in both these cases, the resultant is a Series object
print(type(market_df['Sales']))
print(type(market_df.Sales))

# Select Cust_id, Sales and Profit:
market_df[['Cust_id', 'Sales', 'Profit']].head()

type(market_df[['Cust_id', 'Sales', 'Profit']])

# Similarly, if you select one column using double square brackets, 
# you'll get a df, not Series

type(market_df[['Sales']])

# Trying to select the third row: Throws an error
market_df[2]

# Changing the row indices to Ord_id
market_df.set_index('Ord_id').head()

market_df.set_index('Order_Quantity').head()

