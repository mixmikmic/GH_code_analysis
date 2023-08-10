import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

df = pd.read_csv('../../clean_data/FinalData_for_Models.csv', index_col=0)

df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize('UTC')
df.index = df.index.tz_convert('America/New_York')

df_count_hourly = df.groupby(df.index.hour).sum()

df.head()

df_count_hourly['num_pickups'].sum()

df_mean_hourly = df.groupby(df.index.hour).mean()

df_mean_hourly.head()

_ = df_mean_hourly.num_pickups.plot()
_ = df_mean_hourly.Passengers.plot()
plt.xlabel('Hour of Day')
plt.ylabel('Mean Count per Hour of Day')
plt.legend(loc='upper left')
plt.title('Mean hourly Incoming Passengers and Pickup Count')
plt.show()

df_daily = df.resample('D').sum()
df_daily.tail()

df_mean_daily = df_daily.groupby(df_daily.index.dayofweek).mean()

df_mean_daily.head()

_ = df_mean_daily.num_pickups.plot()
_ = df_mean_daily.Passengers.plot()
plt.xlabel('Day of Week')
plt.ylabel('Mean Count per Hour of Day')
plt.legend(loc='upper right')
plt.title('Mean Incoming Passengers and Pickup Count vs Day of Week')
plt.show()

df_bar_plot = df_mean_daily.copy()

df_bar_plot.rename(index = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}, inplace=True)

df_bar_plot = df_bar_plot[['num_pickups', 'Passengers']]
ax = df_bar_plot.plot(kind='bar', figsize=(15,4), fontsize='16')
ax.legend(['Pickup Count', 'Passengers'], fontsize='12')
plt.title('Daily Pickups and Incoming Passengers', fontsize='18')

df2 = df.groupby([df.index.dayofweek, df.index.hour])['num_pickups', 'Passengers'].mean()

df3 = pd.DataFrame(df2)

df3.reset_index(inplace=True)

df4 = df3.groupby([df3.level_0 < 6, df3.level_1])['num_pickups', 'Passengers'].mean()

df4.index.set_levels([['Weekend', 'Weekday'], range(24)], inplace=True)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4), sharey=True)

df4.loc['Weekday'].plot(ax=axes[0], fontsize='14')
axes[0].set_title('Weekday', fontsize='16')
axes[0].set_xlabel('Hour', fontsize='16')
axes[0].set_ylabel('Number', fontsize='16')
axes[0].legend(['Pickup Count', 'Passengers'], fontsize='12')

df4.loc['Weekend'].plot(ax=axes[1], fontsize='14')
axes[1].set_title('Weekend', fontsize='16')
axes[1].set_xlabel('Hour', fontsize='16')
axes[1].legend(['Pickup Count', 'Passengers'], fontsize='12')
fig.suptitle('Average Number of Incoming Passengers and Pickups vs Hour of Day', y=1.05, fontsize='20')



