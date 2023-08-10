from stock_utils import *

df = pd.DataFrame()
df = df.from_csv('stock_data/tsla.csv')

## Get movement data
movement = get_price_movement_percentages(df)
weekly_movement = get_price_movement_percentages(df, period=7)
monthly_movement = get_price_movement_percentages(df, period=30)

plt.figure(figsize=(10,4))
plot_gaussian_categorical(movement, title='TSLA, Daily')
plt.figure(figsize=(10,4))
plot_gaussian_categorical(weekly_movement, title='TSLA, Weekly', x_min = min(weekly_movement), x_max= max(weekly_movement))
plt.figure(figsize=(10,4))
plot_gaussian_categorical(monthly_movement, title='TSLA, Monthly', x_min = min(monthly_movement), x_max= max(monthly_movement))
plt.show()

## Break up into two groups so we don't create two many plots at once
g = glob.glob('stock_data/*.csv')

g1 = g[0:int(len(g)/2)]
g2 = g[int(len(g)/2):len(g)]

periods = [1, 7, 30]

## First Group
for i in range(len(g1)):
    df = pd.DataFrame()
    df = df.from_csv(g1[i])
    for t in periods:
        plt.figure(figsize=(10,4))
        movement = get_price_movement_percentages(df, period=t)
        ticker = ticker_from_csv(g1[i])
        ticker_period = ticker + ', ' + str(t)
        plot_gaussian_categorical(movement, title=ticker_period, x_min = min(movement), x_max= max(movement))
        
plt.show()

## Second Group
for i in range(len(g2)):
    df = pd.DataFrame()
    df = df.from_csv(g2[i])
    for t in periods:
        plt.figure(figsize=(10,4))
        movement = get_price_movement_percentages(df, period=t)
        ticker = ticker_from_csv(g2[i])
        ticker_period = ticker + ', ' + str(t)
        plot_gaussian_categorical(movement, title=ticker_period, x_min = min(movement), x_max= max(movement))
        
plt.show()

