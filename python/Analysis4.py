import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().magic('matplotlib inline')

extracted_files = glob.glob('..\\Data\\*\\*\\processed\\*.csv')
df = pd.DataFrame()
list_ = []
for fileName in extracted_files:
    short_df = pd.read_csv(fileName)
    list_.append(short_df)
df = pd.concat(list_)

df.head()

df = df[df['payment_type'].isin([1,2])]

#Ignoring the outliars
df = df[df['tip_amount']<df['fare_amount']]
df = df[df['total_amount']<150]
df = df[df['pickup_area'] != 'Not Specified']
df = df[df['dropoff_area'] != 'Not Specified']
df

df.head()

df['payment_type'] = df.payment_type.apply(lambda x: 'Credit' if x == 1 else 'Cash')
df = df.rename(columns = {'dropoff_area':'Dropoff Area', 'payment_type':'Payment Type', 'total_amount':'Amount'})

grouped_series_df = df['Amount'].groupby([df['Dropoff Area'], df['Payment Type']]).mean()
grouped_df = grouped_series_df.to_frame()
grouped_df = grouped_df.reset_index()
grouped_df = grouped_df.round(2)
grouped_df

plt.subplots(figsize=(20,10))
rc={'font.size': 24, 'axes.labelsize': 24, 'legend.fontsize': 24.0, 
    'axes.titlesize': 32, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
sns.set(rc=rc, style='whitegrid')
violin_plot = sns.violinplot(x="Dropoff Area", y="Amount", hue="Payment Type", data=df, split=True,
                             inner="quart", palette={"Cash": "#DC143C", "Credit": "#FF8C00"})
violin_plot.set_ylim(-25, 200)

#title
violin_plot.set_title('Cash Vs. Credit Distribution For Trips Under $150')
violin_plot.title.set_fontsize(36)
violin_plot.title.set_position([.5, 1.02])
violin_plot.title.set_fontweight(weight='bold')
violin_plot.title.set_color('#4D4D4D')

#XyLabels
violin_plot.xaxis.get_label().set_fontsize(30)
violin_plot.xaxis.get_label().set_fontweight(weight='bold')
violin_plot.xaxis.get_label().set_color('#4D4D4D')

violin_plot.yaxis.get_label().set_fontsize(30)
violin_plot.yaxis.get_label().set_fontweight(weight='bold')
violin_plot.yaxis.get_label().set_color('#4D4D4D')

#Ticks
plt.setp(violin_plot.get_xticklabels(), rotation=45, fontsize=24, color='#B2912F', fontweight='bold')
plt.setp(violin_plot.get_yticklabels(), fontsize=24, color='#B2912F', fontweight='bold')

violin_plot.set(xlabel='Dropoff Area', ylabel='Trip Amount')

sns.despine(left=True)

