# Loading libraries and files
import numpy as np
import pandas as pd

market_df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
customer_df = pd.read_csv("../Data/global_sales_data/cust_dimen.csv")
product_df = pd.read_csv("../Data/global_sales_data/prod_dimen.csv")
shipping_df = pd.read_csv("../Data/global_sales_data/shipping_dimen.csv")
orders_df = pd.read_csv("../Data/global_sales_data/orders_dimen.csv")

# Merging the dataframes one by one
df_1 = pd.merge(market_df, customer_df, how='inner', on='Cust_id')
df_2 = pd.merge(df_1, product_df, how='inner', on='Prod_id')
df_3 = pd.merge(df_2, shipping_df, how='inner', on='Ship_id')
master_df = pd.merge(df_3, orders_df, how='inner', on='Ord_id')

master_df.head()

# Which customer segments are the least profitable? 

# Step 1. Grouping: First, we will group the dataframe by customer segments
df_by_segment = master_df.groupby('Customer_Segment')
df_by_segment

# Step 2. Applying a function
# We can choose aggregate functions such as sum, mean, median, etc.
df_by_segment['Profit'].sum()

# Alternatively
df_by_segment.Profit.sum()

# For better readability, you may want to sort the summarised series:
df_by_segment.Profit.sum().sort_values(ascending = False)

# Converting to a df
pd.DataFrame(df_by_segment['Profit'].sum())

# Let's go through some more examples
# E.g.: Which product categories are the least profitable?

# 1. Group by product category
by_product_cat = master_df.groupby('Product_Category')

# 2. This time, let's compare average profits
# Apply mean() on Profit
by_product_cat['Profit'].mean()

# E.g.: Which product categories and sub-categories are the least profitable?
# 1. Group by category and sub-category
by_product_cat_subcat = master_df.groupby(['Product_Category', 'Product_Sub_Category'])
by_product_cat_subcat['Profit'].mean()

# Recall the df.describe() method?
# To apply multiple functions simultaneously, you can use the describe() function on the grouped df object
by_product_cat['Profit'].describe()

# Some other summary functions to apply on groups
by_product_cat['Profit'].count()

by_product_cat['Profit'].min()

# E.g. Customers in which geographic region are the least profitable?
master_df.groupby('Region').Profit.mean()

# Note that the resulting object is a Series, thus you can perform vectorised computations on them

# E.g. Calculate the Sales across each region as a percentage of total Sales
# You can divide the entire series by a number (total sales) easily 
(master_df.groupby('Region').Sales.sum() / sum(master_df['Sales']))*100

