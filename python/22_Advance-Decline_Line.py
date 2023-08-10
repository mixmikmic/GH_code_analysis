from stock_utils import *

##
## NOTE: This relies on all the data in this folder being up to date, as it will
##       look back some specified number of days. So if the stock data ends at different days,
##       it will misrepresent the actual values we are looking for
g = glob.glob('new_stock_data/*.csv')

viewing_window = 50

advances_for_this_window = []
declines_for_this_window = []

for i in range(-viewing_window, 0):
    advances_today = []
    declines_today = []
    for j in range(len(g)):
        df = pd.DataFrame()
        df = df.from_csv(g[j])
        df = df.sort_index(axis=0)
        
        ticker = ticker_from_csv(g[j])
        
        ## For CSV's lacking data
        if not len(df) - 1 > viewing_window:
            continue
        
        #print((ticker,i))
        close_difference = df['close'][i] - df['close'][i-1]
        if close_difference > 0:
            advances_today.append((ticker, np.absolute(close_difference)))
        elif close_difference < 0:
            declines_today.append((ticker, np.absolute(close_difference)))
            
    advances_for_this_window.append(advances_today)
    declines_for_this_window.append(declines_today)

for i in range(-viewing_window, 0):
    print('Date is today ' + str(i))
    print('Number of advances on this day was: ' + str(len(advances_for_this_window[i])))
    print('Number of declines on this day was: ' + str(len(declines_for_this_window[i])))
    print()

SPY = pd.DataFrame()
SPY = SPY.from_csv('stock_data/spy.csv')
SPY = get_close_price(SPY)

SPY[-viewing_window:]

ad_line = np.zeros(len(advances_for_this_window))
current_ad = 0

for i in range(len(advances_for_this_window)):
    current_ad += len(advances_for_this_window[i]) - len(declines_for_this_window[i])
    ad_line[i] = current_ad
    
plt.figure(figsize=(16,6))
plt.plot(ad_line, label='A/D Line')
plt.title('Advance-decline line and SPY for the past ' + str(viewing_window) + ' days')
#plt.figure(figsize=(16,6))
plt.plot(SPY[-viewing_window:], label='SPY index')
plt.legend()
plt.show()

divergence = []
for i in range(viewing_window):
    divergence.append(ad_line[i] - SPY[-viewing_window + i])

plt.figure(figsize=(16,6))
plt.title('Divergence between the index and the indicator')
plt.plot(divergence)
plt.show()



