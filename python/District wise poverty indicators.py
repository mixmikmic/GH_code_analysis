import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv ('Dataset/District wise Poverty Indicators, 2011.csv')

df.head()

df['Measure Names'].unique()

poor_df = df[df['Measure Names'] == 'Number of poor']

poor_df['Measure Names'].unique()

poor_df = poor_df.drop_duplicates('District Name')

central_df = poor_df[poor_df['Development Region'] == 'Central']
far_western_df = poor_df[poor_df['Development Region'] == 'Far-Western']
western_df = poor_df[poor_df['Development Region'] == 'Western']
mid_western_df = poor_df[poor_df['Development Region'] == 'Mid-Western']
eastern_df = poor_df[poor_df['Development Region'] == 'Eastern']

central_df

plt.figure(figsize=(28,10))
sns.set_context('poster',font_scale=1)
sns_plot = sns.barplot(x='District Name',y='Measure Values',data=central_df)
plt.title('No. of poor')
plt.xlabel('Districts')
plt.ylabel('Measure Values')
plt.tight_layout()

fig =sns_plot.get_figure()
fig.savefig('Charts/Number of poor in Central Development Region.png',dpi=500)







