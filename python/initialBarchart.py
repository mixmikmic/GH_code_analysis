#Imports
get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# reupload all data to have as one dataset again
completedf1 = pd.concat([pd.read_csv('0204plays.csv'),pd.read_csv('0507plays.csv'),pd.read_csv('0810plays.csv'),pd.read_csv('1114plays.csv')])

#(remove 2002, because it has served it's purpose of creating 2003 priors)
df = completedf1[completedf1['year']>2002].reset_index(drop=True)

titlesr = []
titlesy = []
titlesp = []
xs = []
rs = []
ys = []
ps = []
sds = []
# iterate through the groupby of the dataframe by down and keep all years 2003-14
for d,df1 in df.groupby('down'):
    titlesr.append('Run Plays by Yards to First Down on Down ' + str(d))
    titlesy.append('Pass Plays by Yards to First Down on Down ' + str(d))
    titlesp.append('Passing % by Yards to First Down on Down ' + str(d))
    # create the unique list of distance to first down
    yds = sorted(df1.yds_to_go.unique())
    xs.append(yds)
    # create lists of the number of run plays, pass plays, and the pass percentage (pass per runs + pass plays) by yds_to_go
    rs.append([int(df1[df1.yds_to_go==y]['isRun'].sum()) for y in yds])
    ys.append([int(df1[df1.yds_to_go==y]['RESP'].sum()) for y in yds])
    ps.append([df1[df1.yds_to_go==y]['RESP'].mean() for y in yds])

# create subplots for bar charts of number of running plays on each down, by yards to go to first down for 2003-14 combined
f, axarr = plt.subplots(2, 2, figsize=(16,14))
axes = axarr[0].tolist()
axes.extend(axarr[1].tolist())

for i in xrange(4):
    axes[i].plot(xs[i], rs[i])
    axes[i].set_title(titlesr[i])
    axes[i].set_ylabel("Plays")
    axes[i].set_xlabel("Yards to Go to First Down")
    axes[i].grid(False)
    sns.despine()

# create subplots for bar charts of number of passing plays on each down, by yards to go to first down
f, axarr = plt.subplots(2, 2, figsize=(16,14))
axes = axarr[0].tolist()
axes.extend(axarr[1].tolist())

for i in xrange(4):
    axes[i].plot(xs[i], ys[i])
    axes[i].set_title(titlesy[i])
    axes[i].set_ylabel("Plays")
    axes[i].set_xlabel("Yards to Go to First Down")
    axes[i].grid(False)
    sns.despine()

# create subplots for bar charts of % of plays that were passes on each down, by yards to go to first down
f, axarr = plt.subplots(2, 2, figsize=(16,14))
axes = axarr[0].tolist()
axes.extend(axarr[1].tolist())

for i in xrange(4):
    axes[i].plot(xs[i], ps[i])
    axes[i].set_title(titlesp[i])
    axes[i].set_ylabel("Passing Percentage")
    axes[i].set_xlabel("Yards to Go to First Down")
    axes[i].grid(False)
    sns.despine()



