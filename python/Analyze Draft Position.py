import pandas as pd
pd.options.display.max_columns=None
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

league_info_df=pd.read_csv("league_info.csv")

league_info_df.describe()

league_info_df[league_info_df['num_teams']==10]

league_standings=pd.read_csv("league_standings-big.csv")

league_standings.describe()

league_info_df['key']=league_info_df.apply(lambda x: str(x['league_id'])+"|"+str(float(x['season'])),axis=1)

league_standings['key']=league_standings.apply(lambda x: str(x['league_id'])+"|"+str(float(x['season'])),axis=1)

ten_man=league_standings[league_standings['key'].isin(league_info_df[league_info_df['num_teams']==10]['key'])]

fig,axes=plt.subplots(2,2,figsize=(15,10))
i=0
j=0
for season, group in ten_man.groupby(["season"]):
    average_points=group.groupby("draft_position").agg([np.mean])
    x=np.arange(1,11)
    axes[j][i].plot(average_points['points_for'])
    axes[j][i].set_ylabel("Average Fantasy Points")
    axes[j][i].set_xlabel("Draft Position")
    axes[j][i].set_xticks(x)
    axes[j][i].set_title("%d NFL Season"%(season))
    i+=1
    if i>1:
        i=0
        j+=1
fig.suptitle("Draft Position vs Fantasy Points (Yahoo 10 Team)", fontsize=18)
plt.subplots_adjust(top=.9)
fig.savefig("Draft Position vs Fantasy Points.PNG")

league_standings.sort_values(["points_for"],ascending=False)

import seaborn as sns

ten_man=ten_man[ten_man['points_for']<2000]#removing outliers

sns.boxplot(y=ten_man['season'],x=ten_man['points_for'],orient="h")

for season, group in ten_man.groupby("season"):
    print(season,group['points_for'].max())
    series=group['points_for']
    ten_man['normed_points_for']=(series - series.mean())/(series.std())#
sns.boxplot(y=ten_man['season'],x=ten_man['normed_points_for'],orient="h")

sns.boxplot(y='draft_position',x='normed_points_for',data=ten_man[abs(ten_man['normed_points_for'])<1], orient="h")

league_standings[league_standings['number_of_moves']>0].plot(x="number_of_moves",y="points_for",kind="scatter")

data=league_standings[(league_standings['number_of_moves']<80)&
                                                                        (league_standings['points_for']<2000)&
                                                                        (league_standings['points_for']>0)]

len(data)


ax=sns.jointplot(x='number_of_moves',y='points_for', data=data[data['number_of_moves']<50].sample(n=3000),
             kind="kde")
ax.set_axis_labels(xlabel='Roster Moves', ylabel='Season Fantasy Points')
plt.savefig("Roster Moves.PNG")

league_standings

plt.save_fig("Roster ")

data['number_of_trades'].describe()


sns.jointplot(x='number_of_trades',y='points_for', data=data.sample(n=3000),
             kind="kde")

league_standings



