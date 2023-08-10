import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import time
import sys
import re
import datetime

# Helper functions
def str_to_int(str):
    str2 = int(str.replace('$','').replace(',',''))
    return str2

def to_currency(int):
    return '{:,}'.format(int)

# def thousands(int):
#     divided = int // 1000
#     return to_currency(divided)

# def y_fmt(x, y):
#     return '{:2.2e}'.format(x).replace('e', 'x10^')

df = pd.read_pickle('data/benson_btcsentiment_df.pkl')
df = df[['BTCPrice','ETHPrice','BTCVol','TxFees','CostperTxn','NoTxns','AvgBlkSz','UniqueAddresses','HashRate','Crypto Market Cap','Nasdaq','Gold','Interest']]
df_all = df
df_hist = df
df.head()



df_all.corr().sort_values('BTCPrice')

# BTC Price vs Google Search Interest
# from matplotlib.ticker import FormatStrFormatter
# import matplotlib.ticker as tick


df_all = df
y1 = pd.Series(df_all['BTCPrice'])
y2 = pd.Series(df_all['Interest'])
x = pd.Series(df_all.index.values)

fig, ax = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(x,y1,'b')
ax2.plot(x,y2,'g')
ax.set_ylabel("Price $US",color='b',fontsize=12)
ax2.set_ylabel("Google Search Interest",color='g',fontsize=12)
ax.grid(True)
plt.title("Bitcoin Price vs. Google Search Interest", fontsize=14)
ax.set_xlabel('Date', fontsize=12)
fig.autofmt_xdate()
# ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

plt.savefig('charts/googlesearchinterest.png')
print(plt.show())



# BTC Price vs Nasdaq
y2 = pd.Series(df_all['Nasdaq'])

fig, ax = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=12,color='blue')
ax2.set_ylabel("Nasdaq Composite Index",fontsize=12,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Nasdaq Composite Index", fontsize=14,color='black')
ax.set_xlabel('Date', fontsize=12, color='black')
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('charts/nasdaq.png')
print(plt.show())



# BTC Price vs Crypto Market Cap
y2 = pd.Series(df_all['Crypto Market Cap'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Crypto Market Cap",fontsize=14,color='green')
# ax.grid(True)
plt.title("Bitcoin Price vs. Crypto Market Cap", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('charts/cryptomarketcap.png')
print(plt.show())

# BTC Price vs ETH Price
y2 = pd.Series(df_all['ETHPrice'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Ethereum Price",fontsize=14,color='green')
# ax.grid(True)
plt.title("Bitcoin Price vs. Ethereum Price", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('charts/ethprice.png')
print(plt.show())

# BTC Price vs ETH Price
y2 = pd.Series(df_all['CostperTxn'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Cost Per Transaction",fontsize=14,color='green')
# ax.grid(True)
plt.title("Bitcoin Price vs. Cost Per Transaction", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('charts/costpertxn.png')
print(plt.show())

# BTC Price vs Volume
df_all = df_all[:365]

y1 = pd.Series(df_all['BTCPrice'])
y2 = pd.Series(df_all['BTCVol'])
x = pd.Series(df_all.index.values)

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Volume",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Volume", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('charts/fig1.png')
print(plt.show())

# BTC Price vs Transaction Fees
y2 = pd.Series(df_all['TxFees'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("TxFees",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Transaction Fees", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('charts/fig2.png')
print(plt.show())

# BTC Price vs Cost per Transaction
y2 = pd.Series(df_all['CostperTxn'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("CostperTxn",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Cost per Transaction", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('charts/fig3.png')
print(plt.show())

# BTC Price vs Number of Transactions
y2 = pd.Series(df_all['NoTxns'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("NumberofTxns",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Number of Transactions", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('charts/fig4.png')
print(plt.show())


# BTC Price vs Block Size
y2 = pd.Series(df_all['AvgBlkSz'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Average Block Size",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Average Block Size", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('charts/fig6.png')
print(plt.show())

# BTC Price vs Unique Addresses
y2 = pd.Series(df_all['UniqueAddresses'])

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Unique Addresses",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Unique Addresses", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('charts/fig7.png')
print(plt.show())



df = df_all
y = pd.Series(df['BTCPrice'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCPriceQuandl.png')
print(plt.show())

df = df[:365]
y = pd.Series(df['BTCPrice'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, LTM",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/LTMBTCPriceQuandl.png')
print(plt.show())

df = df[:90]
y = pd.Series(df['BTCPrice'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, Last 90 Days",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/90DBTCPriceQuandl.png')
print(plt.show())



df = df_all
y = pd.Series(df['TxFees'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Fees (USD), Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Fees', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCTxnFeesQuandl.png')
plt.show()

df = df[:365]
y = pd.Series(df['TxFees'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Fees (USD), LTM",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Fees', fontsize=12)
plt.tight_layout()
plt.savefig('charts/LTMBTCTxnFeesQuandl.png')
plt.show()

df = df[:90]
y = pd.Series(df['TxFees'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Fees (USD), Last 90 Days",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Fees', fontsize=12)
plt.tight_layout()
plt.savefig('charts/90DBTCTxnFeesQuandl.png')
plt.show()

df = df_all
y = pd.Series(df['CostperTxn'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Cost Per Transaction (BTC?), Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cost per Transaction', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCCostperTxnQuandl.png')
plt.show()

df = df_all
y = pd.Series(df['BTCVol'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Volume (USD), Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCTxnVolQuandl.png')
plt.show()

df = df_hist
y = pd.Series(df['NoTxns'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Number of Bitcoin Transactions, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCTxnAmtQuandl.png')
plt.show()



df = df_all
y = pd.Series(df['AvgBlkSz'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Average Block Size, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Block Size', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCAvgBlockSizeQuandl.png')
plt.show()

# Quandl: Unique BTC Addresses
df = df_all
y = pd.Series(df['UniqueAddresses'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Number of Unique Bitcoin Addresses, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Unique Addresses', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCNoAddressesQuandl.png')
plt.show()



# Quandl: BTC Hash Rate
df = df_all
y = pd.Series(df['HashRate'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Hash Rate, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Hash Rate', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistBTCHashRateQuandl.png')
plt.show()



btcsa = pd.read_pickle('data/benson_btcsentiment_dfs.pkl')

y1 = pd.Series(btcsa['BTC']).apply(str_to_int)
y2 = pd.Series(btcsa['Bitcoin_S'])
x = pd.Series(btcsa.index.values)

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Market Cap",fontsize=14,color='blue')
ax2.set_ylabel("#bitcoin Sentiment",fontsize=14,color='green')
# ax.grid(True)
plt.title("Bitcoin Sentiment Analysis", fontsize=20,color='black')
ax.set_xlabel('Time', fontsize=14, color='black')
# plt.yticks(np.arange(0,max(y1),1e11))
plt.savefig('charts/btcsa.png')
print(plt.show())

y1 = pd.Series(btcsa['ETH']).apply(str_to_int)
y2 = pd.Series(btcsa['Ethereum_S'])
x = pd.Series(btcsa.index.values)

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Market Cap",fontsize=14,color='blue')
ax2.set_ylabel("#ethereum Sentiment",fontsize=14,color='green')
# ax.grid(True)
plt.title("Ethereum Sentiment Analysis", fontsize=20,color='black')
ax.set_xlabel('Time', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('charts/ethsa.png')
print(plt.show())

y1 = pd.Series(btcsa['Crypto Market Cap']).apply(str_to_int)
y2 = pd.Series(btcsa['Blockchain_S'])
x = pd.Series(btcsa.index.values)

ax = plt.gca()
ax2 = ax.twinx()
plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Market Cap",fontsize=14,color='blue')
ax2.set_ylabel("#blockchain Sentiment",fontsize=14,color='green')
# ax.grid(True)
plt.title("Blockchain Sentiment Analysis", fontsize=20,color='black')
ax.set_xlabel('Time', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('charts/blksa.png')
print(plt.show())

df = pd.read_pickle('data/benson_btcsentiment_dfts.pkl')

y = pd.Series(df['Close'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Historical Bitcoin Closing Price",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistoricalBitcoinPrice.png')
print(plt.show())

# Bitcoin Closing Price, past year
df = df[:365]
y = pd.Series(df['Close'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, LTM",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/BitcoinPrice2017.png')
print(plt.show())

# Bitcoin past 90 days
df = df[:90]
y = pd.Series(df['Close'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, past 90 days",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/BitcoinPrice2017.png')
plt.show()

# Coinmarketcap scraping: Ethereum
df_eth = pd.read_pickle('data/benson_btcsentiment_dftseth.pkl')
df_eth.head()

y = pd.Series(df_eth['Close'])
x = pd.Series(df_eth.index.values)

plt.plot(x,y)
plt.title("Historical Ethereum Closing Price",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/HistoricalEthereumPrice.png')
print(plt.show())

# LTM
df_eth = df_eth[:365]
# print(df.tail())
y = pd.Series(df_eth['Close'])
x = pd.Series(df_eth.index.values)

plt.plot(x,y)
plt.title("Ethereum Closing Price Since 2017",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/EthereumPrice2017.png')
print(plt.show())

# Last 90 days
df_eth = df_eth[:365]
# print(df.tail())
y = pd.Series(df_eth['Close'])
x = pd.Series(df_eth.index.values)

plt.plot(x,y)
plt.title("Ethereum Closing Price Since 2017",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('charts/EthereumPrice2017a.png')
plt.show()



