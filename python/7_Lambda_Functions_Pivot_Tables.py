# Loading libraries and files
import numpy as np
import pandas as pd

market_df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
customer_df = pd.read_csv("../Data/global_sales_data/cust_dimen.csv")
product_df = pd.read_csv("../Data/global_sales_data/prod_dimen.csv")
shipping_df = pd.read_csv("../Data/global_sales_data/shipping_dimen.csv")
orders_df = pd.read_csv("../Data/global_sales_data/orders_dimen.csv")

# Merging the dataframes to create a master_df
df_1 = pd.merge(market_df, customer_df, how='inner', on='Cust_id')
df_2 = pd.merge(df_1, product_df, how='inner', on='Prod_id')
df_3 = pd.merge(df_2, shipping_df, how='inner', on='Ship_id')
master_df = pd.merge(df_3, orders_df, how='inner', on='Ord_id')

master_df.head()

# Create a function to be applied
def is_positive(x):
    return x > 0

# Create a new column
master_df['is_profitable'] = master_df['Profit'].apply(is_positive)
master_df.head()

# Create a new column using a lambda function
master_df['is_profitable'] = master_df['Profit'].apply(lambda x: x > 0)
master_df.head()

# Comparing percentage of profitable orders across customer segments
by_segment = master_df.groupby('Customer_Segment')
by_segment.is_profitable.mean()

# Comparing percentage of profitable orders across product categories
by_category = master_df.groupby('Product_Category')
by_category.is_profitable.mean()

# You can also use apply and lambda to alter existing columns
# E.g. you want to see Profit as one decimal place
# apply the round() function 
master_df['Profit'] = master_df['Profit'].apply(lambda x: round(x, 1))
master_df.head()

# Creating a column Profit / Order_Quantity
master_df['profit_per_qty'] = master_df['Profit'] / master_df['Order_Quantity']
master_df.head()

# Read documentation
help(pd.DataFrame.pivot_table)

# E.g. Compare average Sales across customer segments
master_df.pivot_table(values = 'Sales', index = 'Customer_Segment', aggfunc = 'mean')

# E.g. compare total number of profitable orders across regions
# Note that since is_profitable is 1/0, we can directly compute the sum
master_df.pivot_table(values = 'is_profitable', index = 'Region', aggfunc = 'sum')

# Grouping by both rows and columns
# Compare the total profit across product categories and customer segments
# Since there are two categorical variables, we use both rows (index) and columns
master_df.pivot_table(values = 'Profit', 
                      index = 'Product_Category', 
                      columns = 'Customer_Segment', 
                      aggfunc = 'sum')

# Computes the mean of all numeric columns across categories
# Notice that the means of Order_IDs are meaningless
master_df.pivot_table(columns = 'Product_Category')

