import numpy as np
import pandas as pd
import holoviews as hv
import asyncio
from queue import Queue
from holoviews.streams import Buffer, Pipe
import streamz
import streamz.dataframe
import ccxt.async as ccxt
hv.extension('bokeh')

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def get_stats(data):
    symb = data[0][0]
    data = [d[1] for d in data]
    mu = np.mean(data)
    sd = np.std(data)
    # Turn data into units of standard deviations from the mean
    data = (data - mu)/sd
    return symb,data
    

symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD']
sources = {}
pipe = Pipe(data=[])
for s in symbols:
    sources[s] = streamz.Stream(stream_name=s)
    sources[s].sliding_window(10).map(get_stats).sink(print)

get_ipython().run_cell_magic('opts', "Bars [width=700, height=500, show_grid=True, xrotation=90, tools=['hover']] {+framewise}", "def barplot(data):\n    data = dict(data)\n    return hv.Bars(data, hv.Dimension('Exchange'), 'deviation from the mean')\nhv.DynamicMap(barplot, streams=[s for s in sources.values()])")

async def poll(tickers):
    i = 0
    kraken = ccxt.kraken()
    while True:
        symbol = tickers[i % len(tickers)]
        yield (symbol, await kraken.fetch_ticker(symbol))
        i += 1
        await asyncio.sleep(kraken.rateLimit / 1000)


async def main():
    async for (symbol, ticker) in poll(symbols): 
        sources[symbol].emit((symbol, ticker['last']))


asyncio.get_event_loop().run_until_complete(main())









