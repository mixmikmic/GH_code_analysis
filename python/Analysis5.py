import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().magic('matplotlib inline')

extracted_files = glob.glob('..\\Data\\*\\*\\lyftdata\\*.csv')
df = pd.DataFrame()
list_ = []
for fileName in extracted_files:
    short_df = pd.read_csv(fileName)
    list_.append(short_df)
df = pd.concat(list_)

#Ignoring the outliars
df = df[df['tip_amount']<df['fare_amount']]
df['passenger_count'] = df['passenger_count'].apply(lambda x: 1 if x == 0 else x)

df.head()

df = df[df['pickup_area'] != 'Not Specified']
df = df[df['dropoff_area'] != 'Not Specified']

df.passenger_count.unique()

def calculate_estimated_lyft_cost(row):
    passenger_count = row['passenger_count']
    #as lyft estimation does not contain tips, substracting tip from Yellow taxi total_amount
    cost_without_tip = row['total_amount'] - row['tip_amount']
    estimated_lyft_cost = 0
    if 0 <= passenger_count <= 1:
        #for 1 passenger lyft line is economical
        estimated_lyft_cost = row['lyft_line_cost']
    elif 2<= passenger_count <=4:
        #for passenger count between 2 to max 4 lyft is economical
        estimated_lyft_cost = row['lyft_cost']
    else:
        #for passenger count between 2 to max 4 lyft is economical
        estimated_lyft_cost = row['lyft_plus_cost']
    
    return pd.Series([cost_without_tip, estimated_lyft_cost])

data =  df.apply(lambda row: calculate_estimated_lyft_cost(row), axis=1)
data.columns = ['cost_without_tip', 'estimated_lyft_cost']
df = df.join(data)

df.head()

grouped_yellow_cab_series_df = df['cost_without_tip'].groupby(df['passenger_count']).mean()
grouped_lyft_series_df = df['estimated_lyft_cost'].groupby(df['passenger_count']).mean()

grouped_yellow_cab_df = grouped_yellow_cab_series_df.to_frame()

grouped_lyft_df = grouped_lyft_series_df.to_frame()

grouped_df = grouped_yellow_cab_df.join(grouped_lyft_df)

grouped_df = grouped_df.reset_index()

grouped_df = grouped_df.round(2)
grouped_df

sns.set_style('whitegrid')
plt.rcParams['xtick.labelsize'] = 24 
plt.rcParams['ytick.labelsize'] = 24 
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 36

x1 = grouped_df['cost_without_tip']
x2 = grouped_df['estimated_lyft_cost']

bar_labels = grouped_df['passenger_count'].unique()

fig = plt.figure(figsize=(20,10))

y_pos = np.arange(len(x1))
y_pos = [x for x in y_pos]
plt.yticks(y_pos, bar_labels, fontsize=24, color='#4D4D4D')

plot1 = plt.barh(y_pos,x1,align='center',color='#FFD700')

plt.barh(y_pos,-x2,align='center',alpha=0.8,color='#FF1493')

# annotation and labels
t = plt.title('Comparison of Lyft Vs Yellow Cab Ride Costs')
plt.xlabel('Ride Cost', color='#4D4D4D', weight='bold')
plt.ylabel('Passenger Count', color='#4D4D4D', weight='bold')
plt.ylim([-1,len(x1)+0.5])
plt.xlim([-max(x2)-5, max(x1)+5])

#Legend
maxbar = plt.Rectangle((1,1),2,2,fc="#FF1493", edgecolor = 'none')
minbar = plt.Rectangle((1,1),2,2,fc='#FFD700',  edgecolor = 'none')
l = plt.legend([maxbar, minbar], ['Lyft', 'Yellow Cab'], loc=2, ncol = 3, prop={'size':24 , 'weight':'bold'})
l.draw_frame(False)

plt.show()

yellow_cost_df = grouped_df[['passenger_count', 'cost_without_tip']]
yellow_cost_df['Cab Type'] = 'Yellow'
yellow_cost_df = yellow_cost_df.rename(columns={'cost_without_tip':'cost'})

yellow_cost_df

lyft_cost_df = grouped_df[['passenger_count', 'estimated_lyft_cost']]
lyft_cost_df['Cab Type'] = 'Lyft'
lyft_cost_df = lyft_cost_df.rename(columns={'estimated_lyft_cost':'cost'})

lyft_cost_df

joined_df = pd.concat([yellow_cost_df,lyft_cost_df])
joined_df

ymax = joined_df.cost.max().round()
ymax

myColors = ["#FFD700","#FF1493"]
rc={'axes.labelsize': 24, 'legend.fontsize': 24.0}
sns.set(rc=rc, style='whitegrid')
factor_plot = sns.factorplot(x="passenger_count", y="cost", hue="Cab Type",
                             data=joined_df, kind = 'bar' , size = 18, palette=myColors,
                            legend=True, legend_out=False)
plt.ylim(0, 18)

#title
figure_title = factor_plot.fig.suptitle('Yellow Cab Vs. Lyft Cost comparison')
figure_title.set_position([.5, 1.02])
figure_title.set_color(color='#4D4D4D')
figure_title.set_fontweight(weight='bold')
figure_title.set_fontsize(36)

#XYLabels
factor_plot.ax.xaxis.set_label_text('Passenger Count', weight='bold')
factor_plot.ax.xaxis.get_label().set_fontsize(30)
factor_plot.ax.xaxis.get_label().set_color('#4D4D4D')

factor_plot.ax.yaxis.set_label_text('Cost', weight='bold')
factor_plot.ax.yaxis.get_label().set_fontsize(30)
factor_plot.ax.yaxis.get_label().set_color('#4D4D4D')

#XTick
factor_plot.set_xticklabels(fontsize=24, color='#B2912F',fontweight='bold')
factor_plot.set_yticklabels(fontsize=24, color='#B2912F',fontweight='bold')

#Annotations
font ={'family': 'serif','weight': 'bold','size': 20, 'color':'#5DA5DA'}
for p in factor_plot.ax.patches:
    percentage = p.get_height()
    factor_plot.ax.text(p.get_x(), percentage+0.1, '%1.2f'%(percentage), fontdict= font)

