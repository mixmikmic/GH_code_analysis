from stock_utils import *

## Get data from csv
df = pd.DataFrame()
df = df.from_csv('stock_data/tsla.csv')

## Get movement data
df['Movement'] = get_price_movement_percentages(df)
x = df['Movement'].as_matrix()

## Show a regular plot for comparison
plot_gaussian(x, fill=True)
plt.show()

def plot_gaussian_categorical(x, x_min=-10, x_max=10, n=10000, title=''):
    ''' 
    Expects an np array of movement percentages, 
    plots the gaussian kernel density estimate
    '''
    ## Learn the kernel-density estimate from the data
    density = stats.gaussian_kde(x)
    
    ## Evaluate the output on some data points
    xs = np.linspace(x_min, x_max, n)
    y = density.evaluate(xs)
    
    ## Create the plot
    plt.plot(xs, y)
    plt.xlabel('Daily Movement Percentage')
    plt.ylabel('Density')
    
    ## Get stats
    mu, sigma = np.mean(x), np.std(x)
 
    ## Plot with conditionals
    plt.fill_between(xs, 0, y, where= xs < mu, facecolor='#eeeedd', interpolate=True) ## Modest Drop
    plt.fill_between(xs, 0, y, where= xs < (mu - sigma / 2), facecolor='yellow', interpolate=True) ## Drop
    plt.fill_between(xs, 0, y, where= xs < (mu - sigma), facecolor='orange', interpolate=True) ## Big Drop
    plt.fill_between(xs, 0, y, where= xs < (mu - 2*sigma), facecolor='red', interpolate=True) ## Very big drop
    
    plt.fill_between(xs, 0, y, where= xs > mu, facecolor='#ebfaeb', interpolate=True) ## Modest Gain
    plt.fill_between(xs, 0, y, where= xs > (mu + sigma/2), facecolor='#b5fbb6', interpolate=True) ## Gain
    plt.fill_between(xs, 0, y, where= xs > (mu + sigma), facecolor='#6efa70', interpolate=True) ## Big Gain
    plt.fill_between(xs, 0, y, where= xs > (mu + 2*sigma), facecolor='green', interpolate=True) ## Very Big Gain
    
    ## Label mu and sigma
    plt.text(x_min + 1, max(y) * 0.8, r'$\mu$ = ' + '{0:.2f}'.format(mu))
    plt.text(x_min + 1, max(y) * 0.9, r'$\sigma$ = ' + '{0:.2f}'.format(sigma))
    ## Set title if given
    if (len(title) != 0):
        plt.title(title)

plt.figure(figsize=(10,4))
plot_gaussian_categorical(x, title='TSLA')
plt.show()

g = glob.glob('stock_data/*.csv')

for i in range(len(g)):
    plt.figure(figsize=(10,4))
    df = pd.DataFrame()
    df = df.from_csv(g[i])
    x = get_price_movement_percentages(df)
    plot_gaussian_categorical(x, title=ticker_from_csv(g[i]))

plt.show()

