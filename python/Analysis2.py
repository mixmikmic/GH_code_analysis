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

df = df[df['pickup_area'] != 'Not Specified']
df = df[df['dropoff_area'] != 'Not Specified']
df

group_by_series = df['total_amount'].groupby([df['pickup_area'], df['payment_type']]).mean()
group_by_df = group_by_series.to_frame()
group_by_df = group_by_df.reset_index()
#Filter Cash and Credit payments
group_by_df = group_by_df[group_by_df['payment_type'].isin([1,2])]
group_by_df['payment_type'] = group_by_df['payment_type'].apply(lambda x: 'Credit' if x == 1 else 'Cash')
group_by_df = group_by_df.rename(columns = {'pickup_area':'Pick-Up Area', 'payment_type':'Payment Type', 'total_amount':'Average'})

group_by_df

myColors = ["#FF8C00","#DC143C"]
rc={'axes.labelsize': 24, 'legend.fontsize': 24.0}
sns.set(rc=rc, style='whitegrid')
factor_plot = sns.factorplot(x="Pick-Up Area", y="Average", hue="Payment Type",
                             data=group_by_df, kind = 'bar' , size = 16, palette=myColors,
                             legend=True, legend_out=False)
factor_plot.despine(left=True)

#Title
figure_title = factor_plot.fig.suptitle('Average Credit Vs. Cash Per Borough')
figure_title.set_position([.5, 1.02])
figure_title.set_color(color='#4D4D4D')
figure_title.set_fontweight(weight='bold')
figure_title.set_fontsize(36)

#XYLabels
factor_plot.ax.xaxis.set_label_text('Borough', weight='bold')
factor_plot.ax.xaxis.get_label().set_fontsize(30)
factor_plot.ax.xaxis.get_label().set_color('#4D4D4D')

factor_plot.ax.yaxis.set_label_text('Total Amount', weight='bold')
factor_plot.ax.yaxis.get_label().set_fontsize(30)
factor_plot.ax.yaxis.get_label().set_color('#4D4D4D')

#XTick
factor_plot.set_xticklabels(rotation=45, fontsize=24, color='#B2912F',fontweight='bold')
factor_plot.set_yticklabels(fontsize=24, color='#B2912F',fontweight='bold')

#Annotations
font ={'family': 'serif','weight': 'bold','size': 20, 'color':'#5DA5DA'}
for p in factor_plot.ax.patches:
    percentage = p.get_height()
    factor_plot.ax.text(p.get_x(), percentage+0.5, '%1.2f'%(percentage), fontdict= font)

