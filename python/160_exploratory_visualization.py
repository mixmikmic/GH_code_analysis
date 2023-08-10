import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plot_prices():
    
    plt.figure(figsize=(12,5))
    df = pd.read_csv("data/GOOG.csv")
    df ['Close'].plot()
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title('GOOG prices at close')
    plt.show()
    
    print "The above graph display GOOG prices at close."
    print "Challenge #1: The data is displayed backwards in time when ready the data directly from the csv file."  
    print "Challenge #2: Only one stock is displayed in the graph. We need to add more stocks for comparison."
    print "Challenge #3: The graph shows 5 years worth of data.  Let's only view the last 2 years in the next graph."
    print '\n'
    print "Let's fix these challenges in the next visual..."
    print '\n'

if __name__ == "__main__":
    plot_prices()

import os

def plot_selected(df, columns, start_index, end_index):

    plot_data(df.ix[start_index:end_index,columns], title="Selected data showing adjusted close price")

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title,fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)

    # Slice and plot
    plot_selected(df, ['SPY','UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD'], '2014-08-01', '2016-08-01')

    print "The above graph displays the list of stocks I've been tracking for a few years."
    print '\n'
    print "Progress from last visual..." 
    print "- Overcame the challenge of showing more than one stock in a graph. The graph is now showing multiple stocks."
    print "- Also, overcame the challenge of creating redundant code by creating a utility function to pull data from each csv file."
    print "- Resolved the challenge of the data displaying stock prices in reverse order."
    print "- Sliced the data to show 2 years worth of data.  The csv files cover 5 years worth of data."
    print '\n'
    print "Challenge: It is hard to visually assess the stocks at different price points."  
    print "This can be resolved by normalizing the data.  Let's fix this challenge in the next visual..." 
    print '\n'

if __name__ == "__main__":
    test_run()

def plot_selected(df, columns, start_index, end_index):
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data showing adjusted close price")
    
    df = df.ix[start_index: end_index, columns]

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Slice and plot
    # plot_selected(df, ['SPY', 'GOOG'], '2014-08-01', '2016-08-01')
    
    # Normalize and plot
    plot_selected_normalize(df, ['SPY','UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD'], '2014-08-01', '2016-08-01')

    print "The above graph normalizes the stock data at a starting point of 1 dollar."
    print '\n'
    print "Result: Whoah!  This graph is way too busy.  The same colour is applied to more than one stock."  
    print "The next series of graphs will divide the stock into smaller groups for more meaningful comparisons..." 
    print '\n'
    
if __name__ == "__main__":
    test_run()

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Normalize and plot
    plot_selected_normalize(df, ['SPY','UPRO'], '2014-08-01', '2016-08-01')
    
    print "- The graph displays the normalized data for SPY and UPRO."
    print "- SPY is an ETF that seeks to provide investment results similar to the S&P 500 index."
    print "- UPRO is an ETF that seeks daily investment results that corresponds to 3x the daily performance of the S&P 500."
    print "\n"
    print "Observations..."
    print "With only a quick overview of this graph, an interesting strategy to execute would be to purchase" 
    print "the UPRO ETF when it is below the SPY ETF. Once purchased, the UPRO stock could then be sold once"
    print "the UPRO ETF appears above the SPY ETF line."
    print "\n"
    
if __name__ == "__main__":
    test_run()

def plot_selected(df, columns, start_index, end_index):
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data showing adjusted close price")
    
    df = df.ix[start_index: end_index, columns]

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Normalize and plot
    plot_selected_normalize(df, ['GOOG','MSFT'], '2014-08-01', '2016-08-01')

    print "- The graph displays the normalized data for GOOG and MSFT."
    print "\n"
    print "Observations..."
    print "With the data normalized, it is amazing to see very similar, sometimes identical, patterns in market behaviour"
    print "starting in August 15 between GOOG and MSFT."
    print "\n"
    
if __name__ == "__main__":
    test_run()

def plot_selected(df, columns, start_index, end_index):
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data showing adjusted close price")
    
    df = df.ix[start_index: end_index, columns]

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Slice and plot
    # plot_selected(df, ['SPY', 'GOOG'], '2014-08-01', '2016-08-01')
    
    # Normalize and plot
    plot_selected_normalize(df, ['AAPL','DIS','NFLX'], '2014-08-01', '2016-08-01')
    
    print "- The graph displays the normalized data for AAPL, DIS and NFLX."
    print "\n"
    print "Observations..."
    print "I own all 3 stocks in my current portfolio."
    print "It is not encouraging to see Disney and Apple barely making a profit in the last two years."
    print "\n"

    
if __name__ == "__main__":
    test_run()

def plot_selected(df, columns, start_index, end_index):
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data showing adjusted close price")
    
    df = df.ix[start_index: end_index, columns]

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Normalize and plot
    plot_selected_normalize(df, ['AMZN','FB','AXY'], '2014-08-01', '2016-08-01')
    
    print "- The graph displays the normalized data for AMZN, FB and AXY."
    print "\n"
    print "Observations..."
    print "I consider these stocks to be real winners for the past 2 years of data."
    print "AMZN looks to have absolutely crushed the competition."
    print "I held some AMZN stock but lost it on an auto-sell during the month of Feb 2016."
    print "I'm still licking my wounds from that auto-sell trigger.  Never should have set that up."
    print "\n"

if __name__ == "__main__":
    test_run()

def plot_selected(df, columns, start_index, end_index):
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data showing adjusted close price")
    
    df = df.ix[start_index: end_index, columns]

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Normalize and plot
    plot_selected_normalize(df, ['SPY','TSLA','GWPH'], '2014-08-01', '2016-08-01')
    
    print "- The graph displays the normalized data for SPY, TSLA and GWPH."
    print "\n"
    print "Observations..."
    print "These stocks haven't really progressed well during the past two years."
    print "The GWPH stocks is showing to be very volatile."
    print "The TSLA stock hasn't been doing to well in the past 2 years."
    print "\n"

    
if __name__ == "__main__":
    test_run()

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="Selected data normalized")
    
    df = df.ix[start_index: end_index, columns]

def symbol_to_path(symbol, base_dir="data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def normalize_data(df):
    return df/ df.ix[0,:]

def plot_data(df, title="Stock prices"):
    ax = df.plot(title=title, fontsize=12,figsize=(12,10))
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def test_run():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['UPRO','GOOG','AAPL','AMZN','DIS','NFLX','FB','AXY','VIX','TSLA','GWPH','MSFT','GLD']  # SPY will be added in get_data()
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Normalize and plot
    plot_selected_normalize(df, ['GLD','VIX'], '2014-08-01', '2016-08-01')
    
    print "- The graph displays the normalized data for GLD and the VIX (volatility index)."
    print "\n"
    print "Observations..."
    print "I was expecting GLD to increase with the VIX but the correlation between the two are weak at best."
    print "\n"
    
if __name__ == "__main__":
    test_run()

