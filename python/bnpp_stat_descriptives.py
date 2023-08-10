import numpy as np
import pandas as pd
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier
import matplotlib
import matplotlib.pyplot as plt
#from mpld3 import enable_notebook # this is a very nice widget for matplotlib
#from mpld3 import plugins
#enable_notebook()

get_ipython().magic('matplotlib inline')

#df = pd.read_csv('/Users/arnaud/cellule/data/bnpp/ETSAnonymousPricesFull.csv', parse_dates=['TradeDateTime'], index_col='TradeDateTime')
df = pd.read_csv('/Users/lecueguillaume/Documents/bnpp/bnpp_donnees/ETSAnonymousPricesFull_2.csv', parse_dates=['TradeDateTime'], index_col='TradeDateTime')

print 'There are {} rows and {} columns'.format(df.shape[0], df.shape[1])

#df[:3]
#print list(df[:1])
#df.values[0]

indices_buy = df['BuySell'] == "Buy"
df2 = df[indices_buy][['Customer', 'Ticker',  'NotionalEUR', 'Sector','TradeStatus']] #'RFQOrderType'
print 'There are {} rows with BuySell = Sell'.format(df2.shape[0])
print 'There are {} tickers and {} clients'.format(len(df2['Ticker'].unique()), len(df2['Customer'].unique()))

df2.head(5)

df2.Sector.unique()

best_tickers = df2['Ticker'].value_counts()[0:30]
best_tickers_notional = df2[['NotionalEUR', 'Ticker']].groupby('Ticker').aggregate(sum)
best_tickers_notional_sort = best_tickers_notional.sort('NotionalEUR', ascending = False)[0:30]
fig, axs = plt.subplots(1,2)
figure_size = (15,5)
best_tickers.plot(ax = axs[0], kind='bar', title = 'Number of RFQ per Ticker', figsize = figure_size)
best_tickers_notional_sort.plot(ax = axs[1], kind = 'bar', title = 'NotionalEUR Quantity traded by Ticker', figsize = figure_size)

print 'The total sum of bounds sold is {} EUR'.format(df2['NotionalEUR'].sum())

tab = df2[['NotionalEUR', 'Ticker']].groupby('Ticker').aggregate(sum).sort('NotionalEUR', ascending = False)
tab.plot(kind='pie', subplots=True, legend=False)
best_tickers_notional = df2[['Ticker', 'NotionalEUR']].groupby('Ticker').aggregate(sum)
best_tickers_notional_sort = best_tickers_notional.sort('NotionalEUR', ascending = False)
nb_tickers_total = len(df2['Ticker'].unique())
nb_tickers = 54
pourcentage =  100*best_tickers_notional_sort[0:nb_tickers].sum()[0]/df2['NotionalEUR'].sum()
print 'There are {0} Tickers (over {2}) representing {1:.2f} per cent of the total number of bounds bought'.format(nb_tickers, pourcentage, nb_tickers_total)
nb_tickers_fin = 500
best_tickers_notional_sort_2 = best_tickers_notional.sort('NotionalEUR')
pourcentage_fin =  100*best_tickers_notional_sort_2[0:nb_tickers_fin].sum()[0]/df2['NotionalEUR'].sum()
print 'There are {0} Tickers (over {2}) representing {1:.2f} per cent of the total number of bounds bought'.format(nb_tickers_fin, pourcentage_fin, nb_tickers_total)

ticker_name_1 = "BNP" # "BACR" #"GAZPRU" # "BACR"
ticker_name_2 = "GAZPRU" #"BNP" # "BACR" # # "BACR"
indices_ticker_1 = df2['Ticker'] == ticker_name_1
indices_ticker_2 = df2['Ticker'] == ticker_name_2
df_ticker_1 = df2[indices_ticker_1]
df_ticker_2 = df2[indices_ticker_2]
fig, axs = plt.subplots(2,1, sharex = True)
df_ticker_1['NotionalEUR'].plot(ax = axs[0], figsize=(15,5), title = 'RFQ on {}'.format(ticker_name_1))
df_ticker_2['NotionalEUR'].plot(ax = axs[1], figsize=(15,5), title = 'RFQ on {}'.format(ticker_name_2))

indices = df2['TradeStatus'] == "Done"
fig, axs = plt.subplots(1,2)
axs[0].set_title('NotionalEUR quantity traded per Sector with TradeStatus = Done')
axs[1].set_title('NotionalEUR quantity traded per Sector')
df2[indices][['NotionalEUR', 'Sector']].groupby('Sector').aggregate(sum).plot(ax = axs[0], kind='bar', figsize=(15,7))
df2[['NotionalEUR', 'Sector']].groupby('Sector').aggregate(sum).plot(ax = axs[1], kind='bar', figsize=(15,7))

indices1 = df2['TradeStatus'] == "Done"
indices2 = df2['Ticker'].isin(["GAZPRU"])
df2[indices1 & indices2][['NotionalEUR']].plot(figsize = [10,5], title = 'RFQ Done on GAZPRU')

best_customers = df2['Customer'].value_counts()[0:30]
best_customers_notional = df2[['Customer', 'NotionalEUR']].groupby('Customer').aggregate(sum)
best_customers_notional_sort = best_customers_notional.sort('NotionalEUR', ascending = False)[0:30]
fig, axs = plt.subplots(1,2)
fig_size = (15,5)
best_customers.plot(ax = axs[0], kind='bar', title = 'number of RFQ per user', figsize = fig_size)
best_customers_notional_sort.plot(ax = axs[1], kind = 'bar', title = 'Quantity traded by user', figsize = fig_size)

tab = df2[['NotionalEUR', 'Customer']].groupby('Customer').aggregate(sum).sort('NotionalEUR', ascending = False)
tab.plot(kind='pie', subplots=True, legend=False, title = 'Repartition of NotionalEUR per users')
best_customers_notional = df2[['Customer', 'NotionalEUR']].groupby('Customer').aggregate(sum)
best_customers_notional_sort = best_customers_notional.sort('NotionalEUR', ascending = False)
nb_cust_total = len(df2['Customer'].unique())
nb_cust = 55
pourcentage =  100*best_customers_notional_sort[0:nb_cust].sum()[0]/df2['NotionalEUR'].sum()
print 'There are {0} customers (over {2}) who bought {1:.2f} per cent of the total number of bounds'.format(nb_cust, pourcentage, nb_cust_total)
nb_cust_fin = 500
best_customers_notional_sort_2 = best_customers_notional.sort('NotionalEUR')
pourcentage_fin =  100*best_customers_notional_sort_2[0:nb_cust_fin].sum()[0]/df2['NotionalEUR'].sum()
print 'There are {0} customers (over {2}) who bought {1:.2f} per cent of the total number of bounds'.format(nb_cust_fin, pourcentage_fin, nb_cust_total)

ind_custom = df2['Customer'] == 1
ind_ticker = df2['Ticker'].isin(["GAZPRU"])
ind_sector = df2['Sector'] == 'Telecom' 
fig, axs = plt.subplots(1,2)
fig_size = (15,5)
df2[ind_custom & ind_sector]['NotionalEUR'].plot(ax = axs[0], figsize = fig_size, title = 'NotionalEUR of Customer 1 in the Telecom Sector')
df2[ind_custom & ind_ticker]['NotionalEUR'].plot(ax = axs[1], figsize = fig_size, title = 'NotionalEUR of Customer 1 for GAZPRU')

id_cust = 1
ind_custom = df2['Customer'] == id_cust
tab = df2[ind_custom][['NotionalEUR', 'Sector']].groupby('Sector').aggregate(sum).sort('NotionalEUR', ascending = False)
tab.plot(kind='pie', subplots=True, legend=False, title = 'NotionalEUR of customer {}'.format(id_cust))

nb_cust_1, nb_cust_2 = 1, 2
ind_custom_1, ind_custom_2 = df2['Customer'] == nb_cust_1, df2['Customer'] == nb_cust_2
fig, axs = plt.subplots(1,2, sharey='row')
title_1 = 'Quantity traded by customer {} per sector'.format(nb_cust_1)
title_2 = 'Quantity traded by customer {} per sector'.format(nb_cust_2)
axs[0].set_title(title_1)
axs[1].set_title(title_2)
fig_size = (15,7)
df2[ind_custom_1][['NotionalEUR', 'Sector']].groupby('Sector').aggregate(sum).plot(ax = axs[0],  kind='bar', figsize=fig_size)#, title = title_1)
df2[ind_custom_2][['NotionalEUR', 'Sector']].groupby('Sector').aggregate(sum).plot(ax = axs[1],  kind='bar', figsize=fig_size)#, title = title_2)

df2['Sector'].unique()

for ele in df2['Sector'].unique():
    ind = df2['Sector'].isin([ele])
    df2[ind][['NotionalEUR', 'Sector']].plot(title = ele)

ind = df2['Sector'] == 'Special Purpose'
df2.ix[ind,'Ticker'].unique()

dff = pd.read_csv('/Users/lecueguillaume/Documents/bnpp/bnpp_donnees/ETSAnonymousPricesFull.csv', parse_dates=['TradeDateTime'])

nb_sectors = 5
ind_buy = dff['BuySell'] == 'Sell'
list_sector = list(dff[ind_buy]['Sector'].value_counts()[0:nb_sectors].keys()) # ['Bank', 'Special Purpose', 'Insurance', 'Telecom', 'Sovereign']
ind_sector = dff['Sector'].isin(list_sector)
dff2 = dff[ind_buy & ind_sector][['TradeDateTime', 'NotionalEUR', 'Sector']]

kw = lambda x: x.isocalendar()[1]; 
kw_year = lambda x: str(x.year) + ' - ' + str(x.isocalendar()[1])
dff2['new_date'] = dff2['TradeDateTime'].map(kw_year)
grouped = dff2.groupby(['new_date', 'Sector'], sort=False, as_index=False).agg({'NotionalEUR': 'sum'})
#grouped.keys()
A = grouped.pivot(index='new_date', columns='Sector', values='NotionalEUR').fillna(0).reset_index()


ticks = A.new_date.values.tolist()
del A['new_date']
ax = A.plot(kind='bar', figsize=(20,10))
ax.set_xticklabels(ticks)
ax.set_title('NotionalEUR per week for the {} best sectors'.format(nb_sectors))
#There is an extra blue sky color???

kw = lambda x: x.isocalendar()[1]; 
kw_year = lambda x: str(x.year) + ' - ' + str(x.isocalendar()[1])
dff2['new_date'] = dff2['TradeDateTime'].map(kw_year)
grouped = dff2.groupby(['new_date', 'Sector'], sort=False, as_index=False).agg({'NotionalEUR': 'sum'})
#grouped.keys()
A = grouped.pivot(index='new_date', columns='Sector', values='NotionalEUR').fillna(0).reset_index()

from mpld3 import enable_notebook # this is a very nice widget for matplotlib
from mpld3 import plugins
enable_notebook()

ticks = A.new_date.values.tolist()
del A['new_date']
ax = A.plot(kind='bar', figsize=(20,10))
ax.set_xticklabels(ticks)
ax.set_title('NotionalEUR per week for the {} best sectors'.format(nb_sectors))



