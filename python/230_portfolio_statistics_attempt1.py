# The code below increases the size of the output screen...

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

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


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12, figsize=(20,5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def compute_daily_returns(df):
    daily_returns = (df/df.shift(1)) - 1
    daily_returns.ix[0,:] = 0
    
    return daily_returns

def compute_cumulative_returns(df, symbols):
    
    for symbol in symbols:
        
        first_price = df[symbol][0]
        last_price = df[symbol][-1]
        cumulative_result = (last_price / first_price - 1) * 100

        print symbol
        print "First price: ", first_price
        print "Last price: ", last_price
        print "cumulative result: ", cumulative_result, "%"
        print "\n"


def run_portfolio_statistics():
    # Read data
    dates = pd.date_range('2015-01-01', '2015-12-31')
    
    # Fictitious portfolio
    symbols = ['SPY','AMZN','FB','AXY','GLD']
    df = get_data(symbols, dates)
    
    # fill empty trade dates (for AXY)
    df.fillna(method="ffill", inplace="True")
    df.fillna(method="bfill", inplace="True")
    
    # plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    
    # Compute cumulative returns
    compute_cumulative_returns(df, symbols)
    
    # Compute average daily returns
    print "Average daily returns: "
    print daily_returns.mean()
    print "\n"
    
    # Compute daily returns - standard deviation
    print "Daily returns - standard deviation: "
    print daily_returns.std()
    print "\n"
    
    # Compute Sharpe ratio
    print "Daily returns - standard deviation: "
    
    print "\n"
    
    
    
    # Scatterplot - SPY and GOOG
    # daily_returns.plot(kind='scatter', x='SPY', y='GOOG')
    # beta_GOOG, alpha_GOOG =np.polyfit(daily_returns['SPY'], daily_returns['GOOG'],1)
    # print "beta_GOOG = ", beta_GOOG, "(Tells you how much more reactive it is to the market than the comparing stock.)"
    # print "alpha_GOOG = ", alpha_GOOG, "(Denotes how well it performs with respect to the comparing stock.)"
    # plt.plot(daily_returns['SPY'], beta_GOOG*daily_returns['SPY'] + alpha_GOOG, '-', color='r')
    # plt.show()
    # print "\n"
    
    # print daily_returns.corr(method='pearson')
    
if __name__ == "__main__":
    
    run_portfolio_statistics()



