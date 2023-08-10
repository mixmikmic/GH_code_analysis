get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')

import numpy as np # numerical computing 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization
import seaborn as sns #modern visualization
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

file_path = 'C:\\Users\\SA31\\Downloads\\'

matches = pd.read_csv(file_path+'matches.csv')

matches.shape

matches.head()

matches.describe()

matches.info()

#matches.shape[0]

matches['id'].max()

matches['season'].unique()

len(matches['season'].unique())

matches.iloc[matches['win_by_runs'].idxmax()]

matches.iloc[matches['win_by_wickets'].idxmax()]

matches.iloc[matches[matches['win_by_runs'].ge(1)].win_by_runs.idxmin()]

matches.iloc[matches[matches['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]

sns.countplot(x='season', data=matches)
plt.show()

#sns.countplot(y='winner', data = matches)
#plt.show

data = matches.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h');

top_players = matches.player_of_match.value_counts()[:10]
#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
#top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v'); #palette="Blues");
plt.show()

ss = matches['toss_winner'] == matches['winner']

ss.groupby(ss).size()

#ss.groupby(ss).size() / ss.count()

#ss.groupby(ss).size() / ss.count() * 100

round(ss.groupby(ss).size() / ss.count() * 100,2)

#sns.countplot(matches['toss_winner'] == matches['winner'])
sns.countplot(ss);

matches[matches['win_by_runs']>0].groupby(['winner'])['win_by_runs'].apply(np.median).sort_values(ascending = False)

#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Runs - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_runs', data=matches[matches['win_by_runs']>0], orient = 'h'); #palette="Blues");
plt.show()

matches[matches['win_by_wickets']>0].groupby(['winner'])['win_by_wickets'].apply(np.median).sort_values(ascending = False)

#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Wickets - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_wickets', data=matches[matches['win_by_wickets']>0], orient = 'h'); #palette="Blues");
plt.show()

