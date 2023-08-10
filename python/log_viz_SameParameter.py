import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('pylab', 'inline')

file_list = ["index_message_2016-6-1_2232.log.csv","index_message_2016-5-30_1729.log.csv", "index_message_2016-5-30_1622.log.csv", "index_message_2016-5-29_2400.log.csv", "index_message_2016-5-29_2327.log.csv", "index_message_2016-4-6_phi10_dh3.log.csv"]
df_list = [pd.read_csv(f, delimiter=';', index_col=0) for f in file_list]
for i, df in enumerate(df_list):
    df.loc[:, 'start_ts'] = df['start_ts'].apply(lambda x: pd.to_datetime(x))
    df.set_index('start_ts', inplace=True)
    df['total_reward'] = df['tr_reward'] - df['op_cost']
    print file_list[i]

plt.figure(1)
ax = plt.subplot(111)
for i, df in enumerate(df_list):
    df['q_sleep'].resample('5Min').plot(figsize=(20, 10), legend=True, ax=ax)

start = pd.to_datetime("2014-09-27 0:00:00")
end = pd.to_datetime("2014-09-28 00:00:00")
delta = pd.Timedelta('2 seconds')
plt.figure(2)
ax = plt.subplot(111)
for i, df in enumerate(df_list):
    step = (df.index-df.index[0])/delta+1
    ts = df['total_reward'].cumsum()/step
    ts.name = file_list[i]
    ts.plot(figsize=(20, 10), legend=True, ax=ax, ylim=(-1, 0))

