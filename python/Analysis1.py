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
df = df.rename(columns = {'pickup_area':'Pick-Up', 'dropoff_area':'Dropoff'})

len(df)

df = df[df['Pick-Up'] != 'Not Specified']
df = df[df['Dropoff'] != 'Not Specified']
df

group_by_series = df['total_amount'].groupby([df['Pick-Up'], df['Dropoff']]).sum()
group_by_df = group_by_series.to_frame()
group_by_df = group_by_df.reset_index()

reshaped = group_by_df.pivot_table('total_amount', 'Pick-Up', 'Dropoff')
reshaped = reshaped.fillna(0.0)
reshaped = reshaped.round(2)

reshaped

plt.subplots(figsize=(20,10))

sns.set_style('whitegrid')
cost_heatmap = sns.heatmap(reshaped, annot=True,
                           annot_kws={"size": 20, "color":'#DAA520' , 'weight':'bold'},
                           linewidths=.5, fmt='.2f', cmap='Blues_r',
                           cbar=True)

#XYTick
plt.setp(cost_heatmap.get_xticklabels(), rotation=45, fontsize=24, color='#B2912F', fontweight='bold')
plt.setp(cost_heatmap.get_yticklabels(), rotation=45, fontsize=24, color='#B2912F', fontweight='bold')

#title
figure_title = cost_heatmap.set_title('Borough to Borough Revenue Distribution', weight='bold')
figure_title.set_position([.5, 1.07])
cost_heatmap.title.set_fontsize(36)
cost_heatmap.title.set_color('#4D4D4D')

#XYLabels
cost_heatmap.xaxis.get_label().set_fontsize(30)
cost_heatmap.xaxis.set_label_text('Dropoff Area', weight='bold')
cost_heatmap.xaxis.get_label().set_color('#4D4D4D')

cost_heatmap.yaxis.get_label().set_fontsize(30)
cost_heatmap.yaxis.set_label_text('Pick-Up Area', weight='bold')
cost_heatmap.yaxis.get_label().set_color('#4D4D4D')

plt.gcf().subplots_adjust(bottom=0.25)

