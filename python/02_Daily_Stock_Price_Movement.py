import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import stats

df = pd.DataFrame()
df = df.from_csv('stock_data/tsla.csv')

df.describe()

def get_price_movements(df):
    movement = np.zeros(len(df))
    last_price = -1
    i = 0
    for index, row in df.iterrows():
        if (last_price < 0):
            movement[i] = 0
        else:
            movement[i] = 100 * row['close'] / last_price - 100
    
        last_price = row['close']
        i += 1
    
    return movement

df['Movement'] = get_price_movements(df.sort_index(axis=0))

plt.figure()
df['Movement'].plot()
plt.show()

def plot_gaussian(x, x_min=-10, x_max=10, n=10000, fill=False, label=''):
    """
    Expects an np array of movement percentages, 
    plots the gaussian kernel density estimate
    """
    ## Learn the kernel-density estimate from the data
    density = stats.gaussian_kde(x)
    
    ## Evaluate the output on some data points
    xs = np.linspace(x_min, x_max, n)
    y = density.evaluate(xs)
    
    ## Create the plot
    if (label != ''):
        plt.plot(xs, y, label=label)
    else:
        plt.plot(xs, y)
    plt.xlabel('Daily Movement Percentage')
    plt.ylabel('Density')
    
    if (fill):
        plt.fill_between(xs, 0, y)

x = df['Movement'].as_matrix()

## Plot the output
plot_gaussian(x, fill=True)
plt.show()

print('The average daily change was ' + str(df['Movement'].mean()) ) 
print('The standard deviation is ' + str(df['Movement'].std()) ) 

g = glob.glob('stock_data/*.csv')

def ticker_from_csv(csv_string):
    stock_name = csv_string.rsplit('.', 1)[0] ## Peel off the trailing ".csv"
    stock_name = stock_name.rsplit('/', 1)[1] ## Peel off the the leading directory name
    return stock_name.upper()

plt.figure()
for i in range(len(g)):
    df = pd.DataFrame()
    df = df.from_csv(g[i])
    ticker = ticker_from_csv(g[i])
    print(ticker)

    df['Movement'] = get_price_movements(df.sort_index(axis=0)) ## Google finance data has dates in descending order, we don't want that
    print('The average daily change was ' + "{0:.2f}".format(df['Movement'].mean()) + ' percent') 
    print('The standard deviation is ' + "{0:.2f}".format(df['Movement'].std()) + ' percent \n' ) 
    df['Movement'].plot()

plt.ylabel('Movement Percentage')
plt.show()

plt.figure()
for i in range(len(g)):
    df = pd.DataFrame()
    df = df.from_csv(g[i])
    df['Movement'] = get_price_movements(df.sort_index(axis=0))
    x = df['Movement'].as_matrix()
    plot_gaussian(x, label=ticker_from_csv(g[i]))

plt.legend()
plt.show()

