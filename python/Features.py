get_ipython().magic('matplotlib inline')
import pandas as pd
import dmc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
matplotlib.style.use('seaborn-notebook')
plt.rcParams['figure.figsize'] = (14.0, 11.0)

data = dmc.data_train()
df = dmc.cleansing.cleanse(data)

returned_articles = df.groupby(['customerID'])['returnQuantity'].sum()
bought_articles = df.groupby(['customerID'])['quantity'].sum()
customer_return_probs = returned_articles / bought_articles

customer_return_probs.plot('hist', bins=[i / 100 for i in range(100)])
None

returned_articles = df.groupby(['productGroup'])['returnQuantity'].sum()
bought_articles = df.groupby(['productGroup'])['quantity'].sum()
group_return_probs = returned_articles / bought_articles

axes = group_return_probs.sort_values().plot('bar')
axes.set_ylim([0,1])
None

returned_articles = df.groupby(['sizeCode']).returnQuantity.sum()
bought_articles = df.groupby(['sizeCode']).quantity.sum()
group_return_probs = returned_articles / bought_articles

axes = group_return_probs.sort_values().plot('bar')
axes.set_ylim([0,1])
None

returned_articles = df.groupby(['colorCode']).returnQuantity.sum()
bought_articles = df.groupby(['colorCode']).quantity.sum()
group_return_probs = returned_articles / bought_articles

axes = group_return_probs.sort_values().plot('bar')
axes.set_ylim([0,1])
None

order_prices = df.groupby(['orderID']).price.sum()
orderShares = df.price / list(order_prices.loc[df.orderID])
orderShares

order_prices = df.groupby(['orderID']).price.sum()
voucher_amounts = df.groupby(['orderID']).voucherAmount.sum()
total_voucher_saving = voucher_amounts.loc[df.orderID] / order_prices.loc[df.orderID]
total_voucher_saving



