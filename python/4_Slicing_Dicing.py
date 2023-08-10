# loading libraries and reading the data
import numpy as np
import pandas as pd

df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
df.head()

# Select all rows where Sales > 3000
# First, we get a boolean array where True corresponds to rows having Sales > 3000
df.Sales > 3000

# Then, we pass this boolean array inside df.loc
df.loc[df.Sales > 3000]

# An alternative to df.Sales is df['Sales]
# You may want to put the : to indicate that you want all columns
# It is more explicit 
df.loc[df['Sales'] > 3000, :]

# We combine multiple conditions using the & operator
# E.g. all orders having 2000 < Sales < 3000 and Profit > 100
df.loc[(df.Sales > 2000) & (df.Sales < 3000) & (df.Profit > 100), :]

# The 'OR' operator is represented by a | (Note that 'or' doesn't work with pandas)
# E.g. all orders having 2000 < Sales  OR Profit > 100
df.loc[(df.Sales > 2000) | (df.Profit > 100), :]

# E.g. all orders having 2000 < Sales < 3000 and Profit > 100
# Also, this time, you only need the Cust_id, Sales and Profit columns
df.loc[(df.Sales > 2000) & (df.Sales < 3000) & (df.Profit > 100), ['Cust_id', 'Sales', 'Profit']]

# You can use the == and != operators 
df.loc[(df.Sales == 4233.15), :]
df.loc[(df.Sales != 1000), :]

# You may want to select rows whose column value is in an iterable
# For instance, say a colleague gives you a list of customer_ids from a certain region

customers_in_bangalore = ['Cust_1798', 'Cust_1519', 'Cust_637', 'Cust_851']

# To get all the orders from these customers, use the isin() function
# It returns a boolean, which you can use to select rows
df.loc[df['Cust_id'].isin(customers_in_bangalore), :]

