# loading libraries and reading the data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set seaborn theme if you prefer
sns.set(style="white")

# read data
market_df = pd.read_csv("../Data/global_sales_data/market_fact.csv")
customer_df = pd.read_csv("../Data/global_sales_data/cust_dimen.csv")
product_df = pd.read_csv("../Data/global_sales_data/prod_dimen.csv")
shipping_df = pd.read_csv("../Data/global_sales_data/shipping_dimen.csv")
orders_df = pd.read_csv("../Data/global_sales_data/orders_dimen.csv")

market_df.head()

# merging with the Orders data to get the Date column
df = pd.merge(market_df, orders_df, how='inner', on='Ord_id')
df.head()

# Now we have the Order_Date in the df
# It is stored as a string (object) currently
df.info()

# Convert Order_Date to datetime type
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# Order_Date is now datetime type
df.info()

# aggregating total sales on each day
time_df = df.groupby('Order_Date')['Sales'].sum()
print(time_df.head())

print(type(time_df))

# time series plot

# figure size
plt.figure(figsize=(16, 8))

# tsplot
sns.tsplot(data=time_df)
plt.show()

# extracting month and year from date

# extract month
df['month'] = df['Order_Date'].dt.month

# extract year
df['year'] = df['Order_Date'].dt.year

df.head()

# grouping by year and month
df_time = df.groupby(["year", "month"]).Sales.mean()
df_time.head()

plt.figure(figsize=(8, 6))
# time series plot
sns.tsplot(df_time)
plt.xlabel("Time")
plt.ylabel("Sales")
plt.show()

# Pivoting the data using 'month' 
year_month = pd.pivot_table(df, values='Sales', index='year', columns='month', aggfunc='mean')
year_month.head()

# figure size
plt.figure(figsize=(12, 8))

# heatmap with a color map of choice
sns.heatmap(year_month, cmap="YlGnBu")
plt.show()

