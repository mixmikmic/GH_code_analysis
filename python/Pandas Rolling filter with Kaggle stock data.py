import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'nbagg')

df = pd.read_csv('all_stocks_5yr.csv', index_col='Date', parse_dates=True)
df.head()

# create a new dataframe with just the apple data
df_aapl = df[df.Name == 'AAPL']

#window length and types
win_len = 10
win_type = [None, 'boxcar', 'triang', 'blackman', 'hamming', 'nuttall']

# plot the raw Open data
ax = df_aapl['Open'].plot(figsize=(7, 4), title='AAPL Open Price')

# loop over the window types
for win in win_type:
    df_aapl_roll = df_aapl.rolling(win_len, win_type=win).mean()
    
    # rename the columns so we can see them in the legend
    df_aapl_roll.columns = [
        '{}_{}_{}'.format(col, win_len, win) for col in df_aapl_roll.columns
    ]
    df_aapl_roll['Open_{}_{}'.format(win_len, win)].plot()

# show everything
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()



