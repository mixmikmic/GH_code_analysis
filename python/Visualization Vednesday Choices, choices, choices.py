import pandas as pd
pd.options.display.max_columns=None
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

league_standings=pd.read_csv("league_standings-small.csv")
league_standings.groupby("draft_position").agg([np.mean, np.median])['points_for'].plot()
plt.ylabel("Average Fantasy Points")
plt.xlabel("Draft Position")
plt.title("Draft Position vs Fantasy Points (Yahoo 10 Team)")

import seaborn as sns
sns.set( palette="pastel", color_codes=True)

sns.boxplot(y=league_standings['draft_position'],x=league_standings['points_for'],orient="h")

sns.boxplot(x=league_standings['draft_position'],y=league_standings['points_for'],orient="v")

sns.violinplot(y=league_standings['draft_position'],x=league_standings['points_for'],inner="quart", orient="h")
sns.despine(left=True)

sns.swarmplot(y=league_standings['draft_position'],x=league_standings['points_for'],orient="h")

league_standings[league_standings['number_of_moves']>0].plot(x="number_of_moves",y="points_for",kind="scatter")

league_standings



