import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_json(path_or_buf="output/user_skill_graph.json", orient="records");

df.value.hist(bins=5)
plt.show()
print df.value.describe()

df.groupby('helperId').mean().hist(bins=5)
plt.show()

df.groupby("timestamp").mean().plot(title="Mean Skill Rating vs Time")
df.groupby("timestamp").median().plot(title="Median Skill Rating vs Time")
plt.show()

df['categories'] = df['categories'].apply(tuple)

new_rows = []
for index, row in df.iterrows():
    new_rows.extend([[row['helperId'], row['timestamp'], row['task'][0], nn, row['value']] for nn in row.categories])
expanded_df = pd.DataFrame(new_rows,columns=['helperId', 'timestamp', 'task', 'category', 'value'])

expanded_df.groupby(['category', 'helperId'])

supply_df = expanded_df[["timestamp", "category", "value"]]
supply_df.set_index('category')
supply_df.loc[:,('Key')] = 'fives'
sdf = supply_df[supply_df.value >= 4]
# supply_df[supply_df.value >= 4].groupby("category").size().plot(kind='bar', color='red')

demand_df = expanded_df[["timestamp", "category", "value"]]
demand_df.set_index('category')
demand_df.loc[:,('Key')] = 'ones'
ddf = demand_df[demand_df.value <= 2]
# demand_df[demand_df.value <= 2].groupby("category").size().plot(kind='bar', color='blue')

DF = pd.concat([sdf,ddf],keys=['fives','ones'])

DFGroup = DF.groupby(['category', 'Key'])
DFGPlot = DFGroup.sum().unstack('Key').plot(kind='bar')
plt.show()

df.categories.map(len).hist(bins=6)
plt.show()
df.categories.map(len).describe()

df.categories.sort_values(ascending=False).iloc[0]

expanded_df.category.value_counts()

categories = list(expanded_df.category.unique())
group_skillset = expanded_df[["timestamp", "category", "value"]]
for category in categories:
    values_to_plot = group_skillset[group_skillset.category == category]
    values_to_plot.groupby("timestamp").mean().plot(title=category)
plt.show()

df['task'] = df['task'].apply(lambda x: x[0])

df[df.value == 1][['task']].ix[:,0].value_counts().head(10)

df[df.value <= 2][['task']].ix[:,0].value_counts().head(10)

df[df.value == 5][['task']].ix[:,0].value_counts().head(10)

df[df.value >= 4][['task']].ix[:,0].value_counts().head(10)

expanded_df.groupby(['helperId', 'category']).mean().plot()
plt.show()





