import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns=1000

# Load data of Chipotle orders
df = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep='\t')
df.head()

# Define items of interest
main_items = ['Bowl', 'Burrito', 'Veggie', 'Chicken Burrito', 'Steak Burrito', 'Chips', 'Bottled Water']

# Determine which orders contain each item of interest
orders = {}
for i in main_items:
    orders[i] = df[df['item_name'].str.contains(i)]['order_id'].unique()
sodas = ['Canned Soft Drink', 'Canned Soda', '6 Pack Soft Drink', 'Izze']
orders['soda'] = df[df['item_name'].isin(sodas)]['order_id'].unique()

# For each order, determine if each is present
df_orders = pd.DataFrame({'order': df['order_id'].unique()})
for i in orders.keys():
    df_orders[i] = df_orders['order'].isin(orders[i])
df_orders.set_index('order', inplace=True)
df_orders.head()

df_plt = df_orders.set_index('Veggie')
df_plt = df_plt.stack().reset_index().rename(columns={'level_1': 'item', 0: 'value'})
df_plt = df_plt[df_plt['item'].isin(['Chips', 'Bottled Water', 'soda'])]
plt.figure(figsize=(5,5))
with sns.plotting_context('poster'):
    sns.barplot(x='item', hue='Veggie', y='value', data=df_plt)
plt.ylabel('Fraction of orders with item')

# Differentiate orders for bowls and burritos
cols_keep = ['Chips', 'Bottled Water', 'soda']
df_bowl = df_orders[df_orders['Bowl']][cols_keep]
df_bowl['Main item'] = 'Bowl'
df_burrito = df_orders[df_orders['Burrito']][cols_keep]
df_burrito['Main item'] = 'Burrito'
df_bb = pd.concat([df_bowl, df_burrito])

# Make figure
df_plt = df_bb.set_index('Main item')
df_plt = df_plt.stack().reset_index().rename(columns={'level_1': 'Side item', 0: 'value'})
df_plt = df_plt[df_plt['Side item'].isin(cols_keep)]
plt.figure(figsize=(5,5))
with sns.plotting_context('poster'):
    sns.barplot(x='Side item', hue='Main item', y='value', data=df_plt)
plt.ylabel('Fraction of orders with side item')

