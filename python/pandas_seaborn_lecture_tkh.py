import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

df = pd.read_csv("../../pandas-seaborn/data/playgolf.csv", delimiter='|')
df.rename(columns=lambda x: x.lower(), inplace=True)
print df.head()

df.info()

df['date'] = pd.to_datetime(df['date'])
df['date']

df2 = df.set_index("date")
print df2.head()

df['temperature'].hist(bins=10, figsize=(8,6));

df.set_index("date",inplace=True)
print df.head()

df['temperature'].plot(figsize=(15,6), marker='o')
plt.title("Temperature Over 2 Weeks", size=20)
plt.xlabel("Time", size=15)
plt.ylabel("Temperature", size=15)
plt.tick_params(labelsize=10);

df[['temperature','humidity']].plot(figsize=(15,6), marker='o')
plt.title("Temperature and Humidity Over 2 Weeks", size=20)
plt.xlabel("Time", size=15)
plt.ylabel("Degrees", size=15);

df[['temperature','humidity']].plot(kind='box',sym='k.', showfliers=True, figsize=(8,6))
plt.title("Boxplots of Temperature and Humidity", size=20)
plt.ylabel("Degrees", size=15)
plt.xlabel("Weather Metrics", size=15);
plt.ylim([0,170])

df[['temperature','humidity']].plot(kind='barh', figsize=(15,10))
plt.title("Comparison of Temperature and Humidity", size=20)
plt.xlabel("Degrees", size=15)
plt.ylabel("Time", size=15);

df2.head()

df2 = df.groupby("outlook").mean()
df2.plot(x=df2.index,y='temperature', kind='barh', figsize=(8,6), legend=None);
plt.title("Mean Temperature Differences For Different Outlooks", size=20)
plt.xlabel("Temperature", size=15)
plt.ylabel("Outlook", size=15);

df.groupby(['outlook','result']).size().plot(kind='bar', figsize=(8,6))
plt.title("Counts by Outlook and Result", size=18)
plt.xlabel("Outlook and Windiness", size=15)
plt.ylabel("Count", size=15)
plt.xticks(rotation=0)
plt.tight_layout();

df.plot(x='temperature',y='humidity', kind='scatter', figsize=(8,6))
plt.title("Scatterplot of Temperature and Humidity", size=18)
plt.xlabel("Temperature", size=15)
plt.ylabel("Outlook", size=15);

play = df['result'] == 'Play'
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
ax[0].scatter(df['temperature'].values, df['humidity'].values, color='k')
ax[0].set_xlabel('Temperature', fontsize=14)
ax[0].set_ylabel('Humidity', fontsize=14)
ax[1].scatter(df['temperature'][play].values, df['humidity'][play].values, 
              color='g', label='Play')
ax[1].scatter(df['temperature'][~play].values, df['humidity'][~play].values, 
              color='r', label="Don't Play")
ax[1].set_xlabel('Temperature', fontsize=14, labelpad=12)
ax[1].legend(loc='best', fontsize=14)
fig.suptitle('Temperature Against Humidity', fontsize=18, y=1.03)
fig.tight_layout(); 

crimes = pd.read_csv("../../pandas-seaborn/data/crime.csv")
print crimes.head()

column_names = crimes.columns[1:]
fig, axs = plt.subplots(4,2, figsize=(10,15))
for i,ax in enumerate(axs.reshape(-1)[:-1]):
    ax.set_title(column_names[i])
    ax.hist(crimes[column_names[i]].values)
fig.delaxes(axs.reshape(-1)[-1])
fig.suptitle("Crime Distributions", y=1.02, fontsize=20)
fig.text(-0.03, 0.5, 'Frequency', fontsize=18, va='center', rotation='vertical')
fig.tight_layout();

crimes.hist(figsize=(15,10));

pd.scatter_matrix(crimes, figsize=(20,20));

sns.set_style('whitegrid')
sns.pairplot(df, hue='result', size=4);

plt.figure(figsize=(8,6))
mvt = sns.distplot(crimes['Motor Vehicle Theft'])

plt.figure(figsize=(8,6))
sns.boxplot(crimes, orient='h');

plt.figure(figsize=(12,10))
sns.heatmap(crimes.corr(), annot=True, linewidth=0.2, cmap='RdYlBu')
plt.tight_layout()

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import bokeh
print 'my bokeh version', bokeh.__version__
output_notebook()

p = figure(plot_width=400, plot_height=400, title="Scatterplot of Temperature and Humidity",title_text_font_size="15pt")

# add a circle renderer with a size, color, and alpha
p.circle(df['temperature'].values, df['humidity'].values, size=5, color="navy", alpha=0.9)
p.yaxis.axis_label = 'Humidity'
p.xaxis.axis_label = 'Temperature'
show(p)

p2 = figure(plot_width=600, plot_height=400, x_axis_type="datetime")

# add a line renderer
p2.line(df.index.values, df['temperature'].values, line_width=2)
p2.yaxis.axis_label = 'Temperature'
p2.xaxis.axis_label = 'Time'
show(p2)

# ! pip install plotly --upgrade

from plotly import __version__
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import Scatter, Layout, XAxis, YAxis
print __version__ # requires version >= 1.9.0
init_notebook_mode()

iplot({"data": [Scatter(x=df.index.date, 
                        y=df['temperature'].values)],
       "layout": Layout(title="Changes Temperature Over Time",
                        xaxis=XAxis(title="Time"),
                        yaxis=YAxis(title="Temperature"))})

iplot({"data": [Scatter(x=df['temperature'].values, 
                        y=df['humidity'].values,
                       mode='markers')],
       "layout": Layout(title="Scatterplot of Temperature and Humidity",
                        xaxis=XAxis(title="Temperature"),
                        yaxis=YAxis(title="Humidity"))})

