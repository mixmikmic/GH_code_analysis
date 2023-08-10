get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

jordan = 'http://espn.go.com/nba/player/stats/_/id/1035/michael-jordan'
lebron = 'http://www.espn.com/nba/player/stats/_/id/1966/lebron-james'

tab1 = pd.read_html(jordan, header=1)
tab2 = pd.read_html(lebron, header=1)

avgs_jordan = tab1[1][0:-1]
avgs_lebron = tab2[1][0:-1]

combined = pd.concat([avgs_jordan['FG%'], avgs_lebron['FG%']],axis='columns')
combined.columns = ['Jordan', 'Lebron']
combined[['Jordan', 'Lebron']] = combined[['Jordan', 'Lebron']].astype(float)

combined

combined.describe()

fig, ax = plt.subplots(figsize=(20, 10))
combined.plot(ax=ax, linewidth=4)
ax.set_title('Season Average Field Goal Percentages', fontsize=30, weight='bold')
ax.set_ylim(bottom=0, top=0.8)
ax.set_xlabel('Number of Seasons')
plt.show()

