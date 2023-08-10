get_ipython().magic('matplotlib inline')
import zipline
import pandas as pd

print "zipline {0}".format(zipline.__version__)
print "pandas {0}".format(pd.__version__)

get_ipython().run_cell_magic('zipline', '--start=2011-1-1 --end=2013-1-1 -o perf', "\nfrom zipline.api import order, record, symbol\nimport matplotlib.pyplot as plt\n\ndef initialize(context):\n    pass\n\ndef handle_data(context, data):\n    order(symbol('AAPL'), 10)\n    record(AAPL=data[symbol('AAPL')].price)\n    \ndef analyze(context, perf):\n    ax1 = plt.subplot(211)\n    perf.portfolio_value.plot(ax=ax1)\n    ax2 = plt.subplot(212, sharex=ax1)\n    perf.AAPL.plot(ax=ax2)\n    plt.gcf().set_size_inches(18, 8)\n    plt.show()")

# zipline + scrapy_giant + pyfolio

get_ipython().magic('matplotlib inline')

# default lib import
import sys
import logbook
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import pytz
import traceback

# pyfolio and zipline import
import pyfolio as pf

from zipline.algorithm import TradingAlgorithm
from zipline.api import (
    FixedSlippage,
    order,
    set_slippage,
    record,
    sid,
)
from zipline.utils.factory import load_from_yahoo
from zipline.finance import commission
from zipline.utils.factory import *
from zipline.finance.trading import SimulationParameters

STOCKS = ['AMD']
start = datetime.utcnow() - timedelta(days=20)
end = datetime.utcnow() - timedelta(days=10)
data = load_from_yahoo(stocks=STOCKS, indexes={}, start=start, end=end)
data = data.dropna()
print data.tail(4)

# Create and run the algorithm by calling func binded
def initialize(algo):
    algo.stock = 'AMD'
    algo.asset = algo.symbol(algo.stock)
    
def handle_data(algo, data):
    algo.order(algo.asset, 1000)
    
def_alg = TradingAlgorithm(handle_data=handle_data, initialize=initialize)
results = def_alg.run(data)
returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(results)
pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

STOCKS = ['2317.TW']
start = datetime.utcnow() - timedelta(days=20)
end = datetime.utcnow() - timedelta(days=10)
data = load_from_yahoo(stocks=STOCKS, indexes={}, start=start, end=end)
data = data.dropna()
print data.tail(4)

# extend basic alg to top
class TestAlgorithm(TradingAlgorithm):
    
    def __init__(self, *args, **kwargs):
        super(TestAlgorithm, self).__init__(args, kwargs)
        self.ssid = kwargs.pop('symbol', '2317.TW')
        self.amount = kwargs.pop('amount', 1000)

    def initialize(self, *args, **kwargs):
        super(TestAlgorithm, self).initialize(args, kwargs)
        self.asset = self.symbol(self.ssid)
    
    def handle_data(self, data):
        super(TestAlgorithm, self).handle_data(data)
        self.order(self.asset, self.amount)

test_alg = TestAlgorithm()
results = test_alg.run(data)
returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(results)
pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

# scrapy_giant import
import talib
import traceback
from IPython.display import display, HTML

from bin.mongodb_driver import *
from bin.start import *

from handler.tasks import collect_hisframe
from handler.hisdb_handler import TwseHisDBHandler, OtcHisDBHandler
from handler.iddb_handler import TwseIdDBHandler, OtcIdDBHandler

from algorithm.report import Report
from algorithm.register import AlgRegister

starttime = datetime.utcnow() - timedelta(days=20)
endtime = datetime.utcnow() - timedelta(days=10)
stockids = ['2317']

# use scrapy_giant as input 
kwargs = {
    'opt': 'twse',
    'targets': ['stock'],
    'starttime': starttime,
    'endtime': endtime,
    'stockids': stockids,
    'traderids': [],
    'base': 'stock',
    'callback': None,
    'limit': 1,
    'debug': True
}
panel, dbhandler = collect_hisframe(**kwargs)
print panel['2317'].tail(4) 

test_alg = TestAlgorithm(**{'symbol':'2317'})
results = test_alg.run(panel)
returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(results)
pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

class DualEMA(TradingAlgorithm):
    """ Dual Moving Average Crossover algorithm
    """

    def __init__(self, dbhandler, *args, **kwargs):
        self._debug = kwargs.pop('debug', False)
        self._cfg = {
            'buf_win': kwargs.pop('buf_win', 30),
            'buy_hold': kwargs.pop('buy_hold', 5),
            'sell_hold': kwargs.pop('sell_hold', 5),
            'buy_amount': kwargs.pop('buy_amount', 1000),
            'sell_amount': kwargs.pop('sell_amount', 1000),
            'short_ema_win': kwargs.pop('short_ema_win', 7),
            'long_ema_win': kwargs.pop('long_ema_win', 20),
            'trend_up': kwargs.pop('trend_up', True),
            'trend_down': kwargs.pop('trend_down', True)
        }
        super(DualEMA, self).__init__(*args, **kwargs)
        self.dbhandler = dbhandler
        self.sids = self.dbhandler.stock.ids
        
    @property
    def cfg(self):
        return self._cfg

    def initialize(self, *args, **kwargs):
        super(DualEMA, self).initialize(args, kwargs)        
        self.asset = self.symbol(self.sids[0]) 
        self.window = deque(maxlen=self._cfg['buf_win'])
        self.invested_buy = False
        self.invested_sell = False
        self.buy = False
        self.sell = False
        self.buy_hold = 0
        self.sell_hold = 0
        
    def handle_data(self, data):
        super(DualEMA, self).handle_data(data)
        self.window.append((
            data[self.asset].open,
            data[self.asset].high,
            data[self.asset].low,
            data[self.asset].close,
            data[self.asset].volume
        ))

        if len(self.window) == self._cfg['buf_win']:
            open, high, low, close, volume = [np.array(i) for i in zip(*self.window)]
            short_ema = talib.EMA(close, timeperiod=self._cfg['short_ema_win'])
            long_ema = talib.EMA(close, timeperiod=self._cfg['long_ema_win'])
            real_obv = talib.OBV(close, np.asarray(volume, dtype='float'))

            self.buy_hold = self.buy_hold - 1 if self.buy_hold > 0 else self.buy_hold
            self.sell_hold = self.sell_hold - 1 if self.sell_hold > 0 else self.sell_hold
            self.buy = False
            self.sell = False

            # sell after buy
            if self._cfg['trend_up']:
                if short_ema[-1] > long_ema[-1] and not self.invested_buy:
                    self.order(self.asset, self._cfg['buy_amount'])
                    self.invested_buy = True
                    self.buy = True
                    self.buy_hold = self._cfg['buy_hold']
                elif self.invested_buy == True and self.buy_hold == 0:
                    self.order(self.asset, -self._cfg['buy_amount'])
                    self.invested_buy = False
                    self.sell = True

            # buy after sell
            if self._cfg['trend_down']:
                if short_ema[-1] < long_ema[-1] and not self.invested_sell:
                    self.order(self.asset, -self._cfg['sell_amount'])
                    self.invested_sell = True
                    self.sell = True
                    self.sell_hold = self._cfg['sell_hold']
                elif self.invested_sell == True  and self.sell_hold == 0:
                    self.order(self.asset, self._cfg['sell_amount'])
                    self.invested_sell = False
                    self.buy = True

            # save to recorder
            signals = {
                'open': open[-1],
                'high': high[-1],
                'low': low[-1],
                'close': close[-1],
                'volume': volume[-1],
                'short_ema': short_ema[-1],
                'long_ema': long_ema[-1],
                'buy': self.buy,
                'sell': self.sell
            }
            self.record(**signals)

# register to alg tasks
AlgRegister.add(DualEMA)

def test_alg_benchmark(opt='twse', debug=True, limit=0):   
    maxlen = 10
    starttime = datetime.utcnow() - timedelta(days=60)
    endtime = datetime.utcnow() - timedelta(days=10)
        
    report = Report(
        'dualema',
        sort=[('buys', False), ('sells', False), ('portfolio_value', False)], limit=20)

    kwargs = {
        'debug': debug,
        'limit': limit,
        'opt': opt
    }
    # fetch all stockids
    idhandler = TwseIdDBHandler(**kwargs)
    stockids = [stockid for stockid in idhandler.stock.get_ids()]
    for stockid in stockids:
        try:
            kwargs = {
                'opt': opt,
                'targets': ['stock'],
                'starttime': starttime,
                'endtime': endtime,
                'stockids': [stockid],
                'traderids': [],
                'base': 'stock',
                'callback': None,
                'limit': 1,
                'debug': debug
            }
            panel, dbhandler = collect_hisframe(**kwargs)
            if len(panel[stockid].index) < maxlen:
                continue
           
            dualema = DualEMA(dbhandler=dbhandler, **{'symbol':stockid})
            results = dualema.run(panel)
            risks = dualema.perf_tracker.handle_simulation_end()  
            report.collect(stockid, results, risks)    
            
        except:
            print traceback.format_exc()
            continue
        
        return report

# run 
report = test_alg_benchmark()
pool = report.pool
results = pool['2317'].dropna()
HTML(results.to_html())

returns, positions, transactions, gross_lev = pf.utils.extract_rets_pos_txn_from_zipline(results)
pf.plot_drawdown_periods(returns, top=5).set_xlabel('Date')

pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                          gross_lev=gross_lev, live_start_date='2016-08-31', round_trips=True)



