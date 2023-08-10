import pandas as pd
import ccxt
import datetime

xchanges = []
for broker in ccxt.exchanges:
    try:
        exchange = getattr(ccxt, broker)()
        market = list(exchange.load_markets().keys())[0]
        exchange.fetch_ohlcv(market)
        print("\033[92m", broker, " Serves OHLCV data")
        xchanges.append(exchange)
    except Exception as e:
        #print("\033[91m", e)
        pass

exchange.describe()['urls']

exchange.describe()['api']

x = ccxt.cex()

t = datetime.datetime(2018,1,1)
for x in xchanges:
    desc = x.describe()
    try:
        url =  desc['urls']['api']  if not 'public' in desc['urls'] else desc['urls']['public'][0]
        url += desc['api'][0] if 'public' not in desc['api'] else desc['api']['public']['get'][0]
        print(url)
    except KeyError as e:
        print(e, desc['urls'], desc['api'])
    except TypeError as e:
        print(e, desc['api'])
#     data = x.fetch_ohlcv('ETH/BTC', since=int(t.timestamp()))
#     df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
#     df.date = pd.to_datetime(df.date)
#     df.set_index('date')
#     df

t.timestamp()

data

import poloniex
start = datetime.datetime(2015,2,1)
while True:
    end = start + datetime.timedelta(days=1)
    try: 
        df = poloniex.get_full_table('USDT_BTC', start, end)
        print("==> Found it: {}".format(start))
        print(df)
        break
    except Exception as e:
        print("tried {}, nothing.".format(start))
        start += datetime.timedelta(days=1)
        continue

get_ipython().magic('pinfo datetime.timedelta')



