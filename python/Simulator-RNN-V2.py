import pandas as pd
import twx
import pandas as pd
import numpy as np
import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# download data
tickers = ['FB', 'GOOG']
days = ['20170412']

data_for_corelation = pd.DataFrame()
for ticker in tickers:
    data_final = pd.DataFrame()
    for day in days:
        try:
            print('{} - {}'.format(ticker, day))

            data = twx.bookquery(
                 day, 
                 ticker, 
                'time: 9:30am 4pm 1s', 
                'direct.mid')
            data_final = data_final.append(data)
        except Exception as e:
            print(str(e))
    data_for_corelation[ticker] = data_final['direct.mid']
    data_for_corelation.to_csv('tech_stock_data_1day.csv')

stock1 = 'FB'
stock2 = 'GOOG'
data = pd.read_csv('tech_stock_data_1day.csv')

def regress(returns1,returns2):
    x = np.asarray(returns1).reshape(-1,1)
    y = np.asarray(returns2).reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)
    a = model.intercept_[0]
    b = model.coef_[0,0]
    residuals = y-model.predict(x)
    return residuals, a,b

def returns(midprices):
    return np.diff(midprices, axis=-1)/midprices[:-1]

l1 = list(data[stock1])
l2 = list(data[stock2])
dataset = regress(l1,l2)[0]

# split into train and test sets
train_size = int(len(dataset) * 0.67)
val_size = int(len(dataset) * 0.10)
test_size = len(dataset) - train_size - val_size

train, val, test = dataset[0:train_size,:],dataset[train_size:train_size+val_size,:],  dataset[train_size+val_size:len(dataset),:]
print(len(train), len(val), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_dataset(train, look_back)
valX, valY = create_dataset(val, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
valX = np.reshape(valX, (valX.shape[0], 1, valX.shape[1]))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(25, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, validation_data=([valX, valY]), epochs=25, batch_size=128, verbose=2)


# make predictions
import sklearn
testPredict = model.predict(testX)
test_mse = sklearn.metrics.mean_squared_error(testY, testPredict)
train_mse = sklearn.metrics.mean_squared_error(trainY, model.predict(trainX))
val_mse = sklearn.metrics.mean_squared_error(valY, model.predict(valX))

print('Train MSE: {}'.format(train_mse))
print('Val MSE: {}'.format(val_mse))
print('Test MSE: {}'.format(test_mse))

# Show that simple prediction works!
arr = np.array([[[2.22424243, 2.2394794 , 2.2394794 , 2.22424243, 2.22424243,
        2.22424243, 2.22424243, 2.28924243, 2.28924243, 2.28924243]]])
model.predict(arr)[0][0]

get_ipython().run_cell_magic('time', '', 'import sys\nfrom simulator import (\n    Simulator, string_to_micro, micro_to_time,\n    BUY, SELL, SHORT, EXCH_INET,\n    BOOK_DEPTH1_PRICE, ORDER_EVENTS,\n    )\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom sklearn.linear_model import LinearRegression\nfrom statsmodels.tsa.ar_model import AR\n\n\nclass Ave_Lee(object):\n    def __init__(self, session, date, tickers, start_time, end_time, model):\n        self.session = session\n        self.date = date\n        self.tickers = tickers\n        self.ticker1 = self.tickers[0]\n        self.ticker2 = self.tickers[1]\n        self.start_time = start_time\n        self.end_time = end_time\n        self.interval = string_to_micro("1s")\n        self.state = \'NULL\'\n        \n        # load RNN Model\n        self.model = model # RNN\n        print(\'test\')\n        arr = np.array([[[2.22424243, 2.2394794 , 2.2394794 , 2.22424243, 2.22424243,\n        2.22424243, 2.22424243, 2.28924243, 2.28924243, 2.28924243]]])\n        print(arr)\n        \n        # try SAME prediction as outside of loop\n        #model.predict(arr)[0][0]\n        print(self.model.predict(arr)[0][0])\n        \n        \n        # variables for BUY or SELL\n        self.side1 = 0\n        self.side2 = 0\n        # variables for order size\n        self.order_size1 = 100\n        self.order_size2 = 1\n        # variables to keep track of total shares bought/sold and the corresponding amount of money\n        self.buy_shares1 = 0\n        self.buy_dollars = 0\n        self.sell_shares1 = 0\n        self.sell_dollars = 0\n        self.buy_shares2 = 0\n        self.sell_shares2 = 0\n        # minimum increment in a bid\n        self.tick_size = 10000\n        \n        # variables to keep track of how many positions we have opened and closed respectively\n        self.open_longs = 0\n        self.open_shorts = 0\n        self.close_longs = 0\n        self.close_shorts = 0 \n        self.runs = 0\n        \n        # variables used for the fitOU, when to open/close a position and how far we look back\n        self.dt = 1\n        self.long_open = 1.25 #1.50 #1.25\n        self.long_close = 0.50 #0.25 #0.50\n        self.short_open = 1.25 #1.50 #1.25;\n        self.short_close = 0.75 #0.50 #0.75\n        self.training_size = 10\n        \n        # start timer/ call the start_callback function\n        self.session.add_timer(self.start_time, self.start_callback)\n        \n        # list to store pnl every time we update it\n        self.pnl = []\n        # dictionary to store time, midprices and the returns each timestep\n        self.results = {\'time\': []}\n        for ticker in self.tickers:\n            self.results[ticker] = []\n            self.results[\'return {}\'.format(ticker)] = []\n        \n        # subscribe to the tickers of interest, and set the first timer\n        for ticker in self.tickers:\n            self.session.subscribe_ticker_all_feeds(ticker)\n    \n    def start_callback(self, time):\n        for ticker in self.tickers:\n            self.session.subscribe_event(ticker, ORDER_EVENTS, self.event_callback)\n        self.session.add_timer(time, self.timer_callback)\n        \n    \n    def event_callback(self, ticker, event_params):\n        # call the execution manager whenever we have an execution\n        self.process_executions(event_params)\n        \n    def timer_callback(self, time):\n        self.runs += 1\n        self.results[\'time\'].append(micro_to_time(time))\n        # get the best bid and offer, compute the midmarket\n        bid1, ask1 = self.session.get_inside_market(self.ticker1)\n        bid2, ask2 = self.session.get_inside_market(self.ticker2)\n        # append the midprices\n        self.results[self.ticker1].append(self.get_midmarket(self.ticker1) / 1000000.0)\n        self.results[self.ticker2].append(self.get_midmarket(self.ticker2) / 1000000.0)\n        \n        # start calculating returns after 1 second\n        if time > self.start_time + 10**6:\n            self.results[\'return {}\'.format(self.ticker1)].append(np.float(self.returns(self.results[self.ticker1][-2:])))\n            self.results[\'return {}\'.format(self.ticker2)].append(np.float(self.returns(self.results[self.ticker2][-2:])))\n        \n        # start collecting signals after training_size * 1 second\n        if time > self.start_time + self.training_size * 10**6:\n            # collect the last training_size of returns\n            returns1 = self.results[\'return {}\'.format(self.ticker1)][-self.training_size:]\n            returns2 = self.results[\'return {}\'.format(self.ticker2)][-self.training_size:]\n            # regress the returns and fit the residuals, calculate the s-score\n            residuals, a,b = self.regress(returns1,returns2)\n            kappa, m, sigma, sigmaeq = self.fitOU(residuals)\n            \n            print(residuals)\n            #arr = np.array([[[1.47795598e-04,  1.77768732e-04,  2.41230408e-05, -6.10925309e-06,\n            #                1.16484998e-04, -4.54248280e-04,  1.35327673e-04, -7.98121981e-05,\n            #                2.77442353e-05, -8.90745451e-05]]])\n            #print(self.model.predict(arr))\n            #s = self.model.predict(np.asarray(residuals).reshape(1,1,10))[0][0]\n            #print(s)\n            \n            s = self.sscore(m, sigmaeq)\n            # find current net position (=0: neutral, <0: we are short asset 1, >0: we are long asset 1)\n            pos = self.buy_shares1 - self.sell_shares1            \n            # feature to check if we have orders at the market before we open a position\n            orders = self.session.get_all_orders()\n        \n            if not orders:\n                if pos == 0:\n                    if s < -self.long_open:\n                        self.side1 = BUY\n                        self.side2 = SELL\n                        price1 = ask1[\'price\']# - self.tick_size\n                        price2 = bid2[\'price\']# + self.tick_size\n                        # make the portfolio self financing by making sure we sell for as much as we buy\n                        self.order_size2 = int(price1 * self.order_size1 / price2)\n                        self.session.add_order(self.ticker1, self.side1, self.order_size1, price1, exchange=EXCH_INET)\n                        self.session.add_order(self.ticker2, self.side2, self.order_size2, price2, exchange=EXCH_INET)\n                        self.open_longs += 1\n                        #print("open long")\n                    elif s > self.short_open:\n                        self.side1 = SELL\n                        self.side2 = BUY\n                        price1 = bid1[\'price\']# + self.tick_size\n                        price2 = ask2[\'price\']# - self.tick_size\n                        # make the portfolio self financing by making sure we buy for as much as we sell\n                        self.order_size2 = int(price1 * self.order_size1 / price2)\n                        self.session.add_order(self.ticker1, self.side1, self.order_size1, price1, exchange=EXCH_INET)\n                        self.session.add_order(self.ticker2, self.side2, self.order_size2, price2, exchange=EXCH_INET)\n                        self.open_shorts += 1\n                        #print("open short")\n                elif pos < 0 and s < self.short_close:\n                    self.side1 = BUY\n                    self.side2 = SELL\n                    price1 = ask1[\'price\']# - self.tick_size\n                    price2 = bid2[\'price\']# + self.tick_size\n                    self.session.add_order(self.ticker1, self.side1, self.order_size1, price1, exchange=EXCH_INET)\n                    self.session.add_order(self.ticker2, self.side2, self.order_size2, price2, exchange=EXCH_INET)\n                    self.close_shorts += 1\n                    #print("short close")\n                elif pos > 0 and s > -self.long_close:\n                    self.side1 = SELL\n                    self.side2 = BUY\n                    price1 = bid1[\'price\']# + self.tick_size\n                    price2 = ask2[\'price\']# - self.tick_size\n                    self.session.add_order(self.ticker1, self.side1, self.order_size1, price1, exchange=EXCH_INET)\n                    self.session.add_order(self.ticker2, self.side2, self.order_size2, price2, exchange=EXCH_INET)\n                    self.close_longs += 1\n                    #print("long close")\n        # update pnl every second to see how it evolves over the day            \n        pnl = self.get_pnl()\n        self.pnl.append(pnl / 1000000.0)\n            \n        # reset the timer unless we are done \n        if time < self.end_time:\n            self.session.add_timer(time + self.interval, self.timer_callback) \n                \n            \n    def process_executions(self, evp):\n        # make sure that we only update if we have executed any orders\n        # when we want to add transaction costs we do it in this function\n        if \'executed_orders\' in evp:\n            time = self.session.current_time()\n            for ex in evp[\'executed_orders\']:\n                order = ex[\'order\']\n                side = order[\'side\']\n                ticker = order[\'ticker\']\n                if ticker == self.ticker1:\n                    if side == \'B\':\n                        self.buy_shares1 += ex[\'quantity_executed\']\n                        #self.buy_dollars += ex[\'quantity_executed\'] * ex[\'price_executed\']\n                        # buy in midmarker to check if spread is "eating" profits\n                        self.buy_dollars += ex[\'quantity_executed\'] * self.get_midmarket(ticker)\n                    else:\n                        self.sell_shares1 += ex[\'quantity_executed\']\n                        #self.sell_dollars += ex[\'quantity_executed\'] * ex[\'price_executed\']\n                        # sell in midmarker to check if spread is "eating" profits\n                        self.sell_dollars += ex[\'quantity_executed\'] * self.get_midmarket(ticker)\n                    pos = self.buy_shares1 - self.sell_shares1\n                elif ticker == self.ticker2:\n                    if side == \'B\':\n                        self.buy_shares2 += ex[\'quantity_executed\']\n                        #self.buy_dollars += ex[\'quantity_executed\'] * ex[\'price_executed\']\n                        # buy in midmarker to check if spread is "eating" profits\n                        self.buy_dollars += ex[\'quantity_executed\'] * self.get_midmarket(ticker)\n                    else:\n                        self.sell_shares2 += ex[\'quantity_executed\']\n                        #self.sell_dollars += ex[\'quantity_executed\'] * ex[\'price_executed\']\n                        # sell in midmarker to check if spread is "eating" profits\n                        self.sell_dollars += ex[\'quantity_executed\'] * self.get_midmarket(ticker)\n                    pos = self.buy_shares2 - self.sell_shares2        \n                pnl = self.get_pnl()\n                #print "{0} {1} {quantity_executed} {price_executed} {liquidity} {2} {3}".format(time, side, pos, pnl, **ex)\n                \n    def create_dataset(dataset, look_back=10):\n        dataX, dataY = [], []\n        for i in range(len(dataset)-look_back-1):\n            a = dataset[i:(i+look_back), 0]\n            dataX.append(a)\n            dataY.append(dataset[i + look_back, 0])\n        return np.array(dataX), np.array(dataY)\n\n    def get_midmarket(self, ticker):\n        bid, ask = self.session.get_inside_market(ticker)\n        return (bid[\'price\'] + ask[\'price\']) / 2.0\n\n    def get_pnl(self):\n        # mark to the midmarket\n        mid1 = self.get_midmarket(self.ticker1)\n        mid2 = self.get_midmarket(self.ticker2)\n        pnl = self.sell_dollars - self.buy_dollars + (self.buy_shares1 - self.sell_shares1) * mid1 + (self.buy_shares2 - self.sell_shares2) * mid2\n        return pnl\n    \n    def regress(self, returns1,returns2):\n        x = np.asarray(returns1).reshape(-1,1)\n        y = np.asarray(returns2).reshape(-1,1)\n        model = LinearRegression()\n        model.fit(x,y)\n        a = model.intercept_[0]\n        b = model.coef_[0,0]\n        residuals = y-model.predict(x)\n        return residuals, a,b\n    \n    def returns(self, midprices):\n        return np.diff(midprices, axis=-1)/midprices[:-1]\n    \n    def fitOU(self, residual):\n        ou = np.cumsum(residual)\n        model = AR(ou)\n        fittedmodel = model.fit(maxlag=1, disp=-1)  \n        a = fittedmodel.params[0]\n        b = fittedmodel.params[1]\n        var =  fittedmodel.sigma2\n        if b > 0.0 and b < np.exp(-2.0/self.training_size):\n            kappa = -np.log(b) / self.dt    \n            m = a / (1.0 - np.exp(-kappa * self.dt))\n            sigma = np.sqrt(var * 2.0 * kappa / (1.0 - np.exp(-2.0 * kappa * self.dt)))\n            sigmaeq = np.sqrt(var / (1.0 - np.exp(-2.0 * kappa * self.dt)));\n            return kappa, m, sigma, sigmaeq\n        else:\n            return -1.0,0,0,0\n    \n    def sscore(self, m, sigmaeq):\n        if sigmaeq != 0:\n            return -m/sigmaeq\n        elif m>0:\n            return 10000000\n        else:\n            return -10000000\n    \n    def end(self):\n        print("Total long opens: " + str(self.open_longs))\n        print("Total short opens: " + str(self.open_shorts))\n        print("Total long closes: " + str(self.close_longs))\n        print("Total short closes: " + str(self.close_shorts))\n        print(\'Runs: \' + str(self.runs))\n        # plot the pnl\n        plt.plot(np.asarray(self.pnl))\n        plt.show()\n        return self.get_pnl()\n\n\'\'\'\ntickers = [\'GOOG\', \'MSFT\', \'AAPL\', \'AMZN\', \'NFLX\', \'CSCO\', \'FB\']\nfor t1 in tickers:\n    for t2 in tickers:\n        if t1 == t2:\n            continue\n        print(\'####\')\n        print(\'{} and {}\'.format(t1, t2))\n        date = "20170413"\n        tickers = [t1, t2]\n        start_time = string_to_micro("9:30")\n        end_time = string_to_micro("10:30")\n        sim = Simulator(Ave_Lee)\n        sim.run(date, tickers, use_om=True, start_time=start_time, end_time=end_time)\n        print(\'####\')\n\'\'\'\n\n\ndate = "20170413"\ntickers = [\'GOOGL\', \'FB\']\nstart_time = string_to_micro("9:30")\nend_time = string_to_micro("10:30")\nsim = Simulator(Ave_Lee)\nsim.run(date, tickers, use_om=True, start_time=start_time, end_time=end_time, model=model)\nprint(\'####\')')



