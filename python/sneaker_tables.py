import pandas as pd
import numpy as np
import pickle

with open('data/final_sales_history.pkl', 'rb') as picklefile:
    sales_history = pickle.load(picklefile)

sales_history.head()

## Add a num_sales (frequency) column
sales_history['num_sales'] = sales_history.groupby('shoe_name')['shoe_name'].transform('count')

## Limit everything below this to shoes with over 50 sales recorded
print(len(sales_history))
sales_history = sales_history[sales_history.num_sales > 50]
print(len(sales_history))

# Creates the shoe_info csv for the flask app
info_cols = ['release_date', 'image_url', 'style_code', 'colorway', 'original_retail',
                          'main_color', 'line', 'brand']

sales_history = sales_history.set_index(['name', 'sale_date_time'])
shoe_info = sales_history[info_cols].reset_index()

# number of transactions in last month
shoe_info = shoe_info[shoe_info.sale_date_time > '2017-02-21']
shoe_info['transactions_last_month'] = shoe_info.groupby('name')['name'].transform('count')

# shoe image URL - remove extra stuff at the end
shoe_info.image_url = shoe_info['image_url'].apply(lambda x: x.split('?')[0])
shoe_info = shoe_info.drop('sale_date_time', 1)
shoe_info = shoe_info.drop_duplicates()
print(shoe_info.head())

# send to CSV
shoe_info.to_csv('data/shoe_info.csv', index = False)

# For the historical sales chart 
chart_data = sales_history.reset_index()
chart_data = chart_data.set_index('sale_date_time')
chart_data['date'] = chart_data.index.date
chart_data = chart_data.reset_index()
chart_data = chart_data[['name', 'sale_date', 'sale_price']].groupby(['name', 'sale_date'])

chart_data = chart_data.aggregate(['count', 'mean', 'min', 'max']).reset_index()
chart_data.columns = chart_data.columns.droplevel(0)
chart_data = pd.DataFrame(chart_data)

# rename columns for clarity
chart_data.columns = ['name', 'date', 'volume', 'sale_mean', 'sale_min', 'sale_max']
chart_data.sale_mean = chart_data['sale_mean'].apply(lambda x: round(x))

# write to CSV
chart_data.to_csv('data/chart_data.csv', index = False)
print(chart_data.head())



