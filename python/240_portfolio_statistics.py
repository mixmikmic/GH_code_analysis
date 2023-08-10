get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plot_selected(df, columns, start_index, end_index):

    plot_data(df.ix[start_index:end_index,columns], title="A portfolio of stocks")

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

def normalize_data(df):
    # normed = price/price[0]
    return df/ df.ix[0,:]

def plot_selected_normalize(df, columns, start_index, end_index):
    
    # Normalize stock prices
    df = normalize_data(df)
    
    plot_data(df.ix[start_index:end_index,columns], title="A portfolio of stocks - normalized")
    
    df = df.ix[start_index: end_index, columns]
    
def compute_daily_returns_portfolio(df):
    daily_returns = (df/df.shift(1)) - 1
    # daily_returns.ix[0,:] = 0
    daily_returns.ix[0] = 0
    
    return daily_returns
    
def run_portfolio_stats():
    # Define a date range
    dates = pd.date_range('2014-08-01', '2016-08-01')

    # Choose stock symbols to read
    symbols = ['SPY','AMZN','FB','AXY','GLD']
    
    # Get stock data
    df = get_data(symbols, dates)
    
    # Fill empty trade dates (for AXY)
    df.fillna(method="ffill", inplace="True")
    df.fillna(method="bfill", inplace="True")

    # Slice and plot
    # plot_selected(df, symbols,'2015-01-01', '2016-01-01')
    
    # Normalize stock prices
    df_normalized = normalize_data(df)
    print "\n"
    print "normalized: "
    print df_normalized.head()
    print "\n"
    
    # Plot normalized data
    # plot_selected_normalize(df, symbols, '2015-01-01', '2016-01-01')
    
    # Reflect fund allocation for each stock
    allocation = [0.2, 0.2, 0.2, 0.2, 0.2]
    df_allocation = df_normalized * allocation
    print "allocated: "
    print df_allocation.head()
    print "\n"
    
    # Reflect starting values for each stock
    # starting_values = [200000, 200000, 200000, 200000, 200000]
    starting_value = [1000000]
    df_with_start_values = df_allocation * starting_value
    print "Show me the money: "
    print df_with_start_values.head()
    print "\n"
    
    # Calculate portfolio value by day
    portfolio_values = df_with_start_values.sum(axis=1)
    print "Portfolio values: "
    print portfolio_values.head()
    print "\n"
    
    # Compute daily returns
    daily_returns_portfolio = compute_daily_returns_portfolio(portfolio_values)
    print "Daily returns: "
    print daily_returns_portfolio.head()
    print "\n"
    
    # Remove first row "0" for portfolio calculations
    daily_returns_portfolio = daily_returns_portfolio[1:]
    # print daily_returns_portfolio.head()
    # print "\n"
    
    # Cumulative return
    print "Cumulative return: ", (portfolio_values[-1] / portfolio_values[0]) - 1
    
    # Average daily return
    print "Average daily return: ", daily_returns_portfolio.mean()

    # Daily standard deviation 
    print "Daily standard deviation: ", daily_returns_portfolio.std()
    
    # Sharpe ratio
    trading_days = 252
    
    Sharpe_ratio = np.sqrt(trading_days) * (daily_returns_portfolio.mean())/daily_returns_portfolio.std()
    print "Sharpe ratio: ", Sharpe_ratio
    print "\n"
    
if __name__ == "__main__":
    run_portfolio_stats()

