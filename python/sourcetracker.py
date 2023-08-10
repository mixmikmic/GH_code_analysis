import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from palettable.colorbrewer.qualitative import Set3_12 as color_palette
import scipy

df = pd.read_csv('./mixing_proportions.txt', sep='\t', header=0, index_col=0)
df.fillna(0, inplace=True)
sns_df = (df*100).round(2)
sns_df.rename(columns={'extreme high temperature habitat':'extreme temperature'}, inplace=True)
sns_df

df_std = pd.read_csv('./mixing_proportions_stds.txt', sep='\t', header=0, index_col=0)
df_std.fillna(0, inplace=True)

def plot_heatmap(dataframe, title=False, savename=False,):
    matplotlib.rcParams.update({'font.size': 36})
    fig, ax = plt.subplots(figsize=(30,40), dpi=50)
    sns.heatmap(ax=ax, 
                annot=False,
                data=dataframe.drop('Unknown', axis=1).T, 
                cbar_kws=dict(use_gridspec=False,location="top", 
                              shrink=1))
    if savename:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()
    

plot_heatmap(sns_df,
            savename='heatmap_perc.png')

plot_heatmap(df_std, 
             title='Standard deviation of mean fractional contribution of each source in sample\n', 
             savename='heatmap_std.png')

def plot_pie_sample(sample, dataframe):
    colors = color_palette.hex_colors

#     crunch the numbers
    
    ser = dataframe.loc[sample]
    ser.sort_values(inplace=True, ascending=False)
    ser.drop(ser.index[11:], inplace=True)
    new = pd.Series(1-ser.sum(), index=['else'])
    ser = ser.append(new)
    
#     plot the graph
    
    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    sources = ser.index
    proportions = ser.values
    w,l,p = ax.pie(proportions, startangle=90, colors=colors, autopct='%1.1f%%',
          pctdistance=1.1, labeldistance=1.3);
    ax.legend(labels=sources, loc='lower left')
    [t.set_rotation(40) for t in p]
    plt.title("Sourcetracker results for sample " + sample)
    plt.savefig('./pie_charts/' + sample + '.png', bbox_inches='tight')
    plt.show()
    

get_ipython().system('mkdir pie_charts')

for samp in df.index:
    plot_pie_sample(samp, df)



