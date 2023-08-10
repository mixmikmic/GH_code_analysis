# loading libraries and reading the data
import numpy as np
import pandas as pd

market_df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
customer_df = pd.read_csv("../Data/global_sales_data/cust_dimen.csv")
product_df = pd.read_csv("../Data/global_sales_data/prod_dimen.csv")
shipping_df = pd.read_csv("../Data/global_sales_data/shipping_dimen.csv")
orders_df = pd.read_csv("../Data/global_sales_data/orders_dimen.csv")

# Already familiar with market data: Each row is an order
market_df.head()

# Customer dimension table: Each row contains metadata about customers
customer_df.head()

# Product dimension table
product_df.head()

# Shipping metadata
shipping_df.head()

# Orders dimension table
orders_df.head()

# Merging the dataframes
# Note that Cust_id is the common column/key, which is provided to the 'on' argument
# how = 'inner' makes sure that only the customer ids present in both dfs are included in the result
df_1 = pd.merge(market_df, customer_df, how='inner', on='Cust_id')
df_1.head()

# Now, you can subset the orders made by customers from 'Corporate' segment
df_1.loc[df_1['Customer_Segment'] == 'CORPORATE', :]

# Example 2: Select all orders from product category = office supplies and from the corporate segment
# We now need to merge the product_df

df_2 = pd.merge(df_1, product_df, how='inner', on='Prod_id')
df_2.head()

# Select all orders from product category = office supplies and from the corporate segment
df_2.loc[(df_2['Product_Category']=='OFFICE SUPPLIES') & (df_2['Customer_Segment']=='CORPORATE'),:]

# Merging shipping_df
df_3 = pd.merge(df_2, shipping_df, how='inner', on='Ship_id')
df_3.shape

# Merging the orders table to create a master df
master_df = pd.merge(df_3, orders_df, how='inner', on='Ord_id')
master_df.shape
master_df.head()

# dataframes having the same columns
df1 = pd.DataFrame({'Name': ['Aman', 'Joy', 'Rashmi', 'Saif'],
                    'Age': ['34', '31', '22', '33'],
                    'Gender': ['M', 'M', 'F', 'M']}
                  )

df2 = pd.DataFrame({'Name': ['Akhil', 'Asha', 'Preeti'],
                    'Age': ['31', '22', '23'],
                    'Gender': ['M', 'F', 'F']}
                  )
df1

df2

# To concatenate them, one on top of the other, you can use pd.concat
# The first argument is a sequence (list) of dataframes
# axis = 0 indicates that we want to concat along the row axis
pd.concat([df1, df2], axis = 0)

# A useful and intuitive alternative to concat along the rows is the append() function
# It concatenates along the rows
df1.append(df2)

df1 = pd.DataFrame({'Name': ['Aman', 'Joy', 'Rashmi', 'Saif'],
                    'Age': ['34', '31', '22', '33'],
                    'Gender': ['M', 'M', 'F', 'M']}
                  )
df1

df2 = pd.DataFrame({'School': ['RK Public', 'JSP', 'Carmel Convent', 'St. Paul'],
                    'Graduation Marks': ['84', '89', '76', '91']}
                  )
df2

# To join the two dataframes, use axis = 1 to indicate joining along the columns axis
# The join is possible because the corresponding rows have the same indices
pd.concat([df1, df2], axis = 1)

