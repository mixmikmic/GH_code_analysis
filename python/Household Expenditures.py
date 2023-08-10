Image('https://cdn2.vox-cdn.com/thumbor/PBFeR5h6uDVbac3CSVzXSpZJkAE=/800x0/filters:no_upscale()/cdn0.vox-cdn.com/uploads/chorus_asset/file/6268393/chart.0.png')

from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import Series, DataFrame

get_ipython().magic('matplotlib inline')

sns.set_style("darkgrid")
plt.style.use('fivethirtyeight')

url = "https://raw.githubusercontent.com/voxmedia/data-projects/master/vox-data/pew-household-expenditures-2016.csv"
raw_data = pd.read_csv(url)
raw_data.head()

raw_data.dtypes

raw_data.columns

# Copy raw_data into a new df
df = raw_data.copy()
df.columns

# Convert "thirds" column to category
df['thirds'] = df['thirds'].astype('category')

df.dtypes

df.columns[2:]

# Save weighted_mean values in a separate dataframe
weighted_mean = df[['weighted_mean_total_expend', 'weighted_mean_food', 'weighted_mean_housing', 'weighted_mean_transportation', 
    'weighted_mean_healthcare', 'weighted_mean_entertainment', 'weighted_mean_apparal', 'weighted_mean_reading',
    'weighted_mean_retirement_pension', 'weighted_mean_cash_contrib', 'weighted_mean_income']].copy()

# We'll be looking at median values for now
df = df[['thirds','median_total_expend', 'median_food','median_housing', 'median_transportation', 'median_healthcare',
         'median_entertainment', 'median_apparal', 'median_reading','median_retirement_pension',
         'median_cash_contrib', 'median_income']]

df

x = range(1996, 2015)
df.index = 3*x

df['housing+food'] = df['median_housing'] + df['median_food']
df['housing+food+transport'] = df['median_housing'] + df['median_food'] + df['median_transportation']
df

# Create three data frames for lower, middle and highest thirds. Reset index and drop old one.
lower = df[df['thirds'] == 'Lower Third'].reset_index(drop=True)
middle = df[df['thirds'] == 'Middle Third'].reset_index(drop=True)
highest = df[df['thirds'] == 'Highest Third'].reset_index(drop=True)

# Set index to years
lower.index = range(1996, 2015)
middle.index = range(1996, 2015)
highest.index = range(1996, 2015)

# ==== Design the figure ==== #

# Set general plot properties
sns.set_context({"figure.figsize": (3, 8)})
sns.set_color_codes(palette='muted')
fig, ax = plt.subplots(figsize=(11, 8))
ax.grid(True,linestyle='-',color='0.9')
ax.set_ylim([0,df['median_income'].max()])
ax.axes.get_xaxis().set_visible(False)
ylim = df['median_income'].max() + 10000

# Bigger font sizes for plots
mpl.rcParams.update({'font.size': 18})
mpl.rc('xtick', labelsize=10) 
mpl.rc('ytick', labelsize=18) 
mpl.rc('labelsize: large')

# ==== Make legend ==== #

leg1 = plt.Rectangle((0,0),1,1,fc="navy", edgecolor = 'none')
leg2 = plt.Rectangle((0,0),1,1,fc='skyblue',  edgecolor = 'none')
leg3 = plt.Rectangle((0,0),1,1,fc='g', edgecolor='none')
leg4 = plt.Rectangle((0,0),1,1,fc='r', edgecolor='none')

l = plt.legend([leg1, leg2, leg3, leg4], ['Housing', 'Food', 'Transportation', 'Income'],
               bbox_to_anchor=(.8,1.01), ncol = 4, prop={'size':16})
l.draw_frame(False)

# Format y axis ticks
def thousands(x, pos):
    'The two args are the value and tick position'
    return '$%iK' % (x*1e-3)
formatter = FuncFormatter(thousands)
ax.yaxis.set_major_formatter(formatter)

# ==== Plot the data ==== #

# Lower plot
ax1=fig.add_subplot(131)
# Bar 1 - background - "total" (top) series
top_plot = sns.barplot(x = lower.index, y = lower['housing+food+transport'], color='g')
# Bar 2 - overlay - "middle" series
middle_plot = sns.barplot(x = lower.index, y = lower['housing+food'], color = "skyblue")
# Bar 3 - overlay - "bottom" series
bottom_plot = sns.barplot(x = lower.index, y = lower['median_housing'], color = "navy")
# Line
income_line = plt.plot(lower['median_income'],'--', color='r')
# Set axes
ax1.set_ylim([0,ylim])
ax1.axes.get_yaxis().set_visible(False)
ax1.set_axis_off()
plt.xticks(rotation=90)

# Middle plot
ax2=fig.add_subplot(132)
top_plot = sns.barplot(x = middle.index, y = middle['housing+food+transport'], color='g')
# Bar 2 - overlay - "middle" series
middle_plot = sns.barplot(x = middle.index, y = middle['housing+food'], color = "skyblue")
# Bar 3 - overlay - "bottom" series
bottom_plot = sns.barplot(x = middle.index, y = middle['median_housing'], color = "navy")
# Line
income_line = plt.plot(middle['median_income'],'--', color='r')
# Set axes
ax2.set_ylim([0,ylim])
ax2.axes.get_yaxis().set_visible(False)
ax2.set_axis_off()
plt.xticks(rotation=90)

# Highest plot
ax3=fig.add_subplot(133)
top_plot = sns.barplot(x = highest.index, y = highest['housing+food+transport'], color='g')
# Bar 2 - overlay - "middle" series
middle_plot = sns.barplot(x = highest.index, y = highest['housing+food'], color = "skyblue")
# Bar 3 - overlay - "bottom" series
bottom_plot = sns.barplot(x = highest.index, y = highest['median_housing'], color = "navy")
# Line
income_line = plt.plot(highest['median_income'],'--', color='r')
# Set axes
ax3.set_ylim([0,ylim])
ax3.axes.get_yaxis().set_visible(False)
ax3.set_axis_off()
plt.xticks(rotation=90)


# ==== Add text ==== #

# Title
title = ax.annotate("Rising rent costs are devastating to low-income families",
            (0,0), (-75, 550), textcoords='offset points', color='gray', fontsize=26, fontweight='heavy')

# Subtitle
ax.annotate("Median incomes and costs of housing, food and transportation across income",
            (0,0), (-75, 525), textcoords='offset points', color='gray', fontsize=16, style='italic')
ax.annotate("groups from 1996 to 2014. Figures adjusted for inflation.",
            (0,0), (-75, 505), textcoords='offset points', color='gray', fontsize=16, style='italic')

# Plot 1 annotations
ax1.annotate("'96", (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points',
             va='top', color='gray')
ax1.annotate("'14", (0,0), (198, -5), xycoords='axes fraction', textcoords='offset points',
             va='top', color='gray')
lower_annotate = ax1.annotate("Lower", (0,0), (70, -20), xycoords='axes fraction', textcoords='offset points',
            va='top', color='gray', fontsize=22, fontweight='bold')

# Plot 2 annotations
ax2.annotate("'96", (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points',
             va='top', color='gray')
ax2.annotate("'14", (0,0), (198, -5), xycoords='axes fraction', textcoords='offset points',
             va='top', color='gray')
ax2.annotate("Middle", (0,0), (70, -20), xycoords='axes fraction', textcoords='offset points',
            va='top', color='gray', fontsize=22, fontweight='bold')

# Plot 3 annotations
ax3.annotate("'96", (0,0), (0, -5), xycoords='axes fraction', textcoords='offset points',
             va='top', color='gray')
ax3.annotate("'14", (0,0), (198, -5), xycoords='axes fraction', textcoords='offset points',
             va='top', color='gray')
ax3.annotate("Upper", (0,0), (70, -20), xycoords='axes fraction', textcoords='offset points',
            va='top', color='gray', fontsize=22, fontweight='bold')

# Source annotations
source_annotate = ax.annotate("Vox Media, March 2016. Data visualization by Will Geary.",
            (0,0), (-75, -75), textcoords='offset points', color='gray', fontsize=14, style='italic')


fig.tight_layout()
fig.savefig('/Users/Will/personal-website/assets/2016-05-26-fig3.png', 
            bbox_extra_artists=(l,title, lower_annotate, source_annotate),
            bbox_inches='tight')



