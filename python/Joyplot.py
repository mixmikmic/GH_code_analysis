from __future__ import unicode_literals
import joypy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

iris = pd.read_csv("data/iris.csv")

get_ipython().magic('matplotlib inline')

fig, axes = joypy.joyplot(iris)

get_ipython().magic('matplotlib inline')
fig, axes = joypy.joyplot(iris, by="Name")

get_ipython().magic('matplotlib inline')
fig, axes = joypy.joyplot(iris, by="Name", ylim='own')

get_ipython().magic('matplotlib inline')
fig, axes = joypy.joyplot(iris, by="Name", overlap=3)

get_ipython().magic('matplotlib inline')
fig, axes = joypy.joyplot(iris, by="Name", column="SepalWidth",
                          hist="True", bins=20, overlap=0,
                          grid=True, legend=False)

get_ipython().magic('matplotlib inline')

temp = pd.read_csv("data/daily_temp.csv",comment="%")
temp.head()

get_ipython().magic('matplotlib inline')

labels=[y if y%10==0 else None for y in list(temp.Year.unique())]
fig, axes = joypy.joyplot(temp, by="Year", column="Anomaly", labels=labels, range_style='own', 
                          grid="y", linewidth=1, legend=False, figsize=(6,5),
                          title="Global daily temperature 1880-2014 \n(°C above 1950-80 average)",
                          colormap=cm.autumn_r)

get_ipython().magic('matplotlib inline')

labels=[y if y%10==0 else None for y in list(temp.Year.unique())]
fig, axes = joypy.joyplot(temp, by="Year", column="Anomaly", labels=labels, range_style='own', 
                          grid="y", linewidth=1, legend=False, fade=True, figsize=(6,5),
                          title="Global daily temperature 1880-2014 \n(°C above 1950-80 average)",
                          kind="counts", bins=30)

get_ipython().magic('matplotlib inline')

fig, axes = joypy.joyplot(temp,by="Year", column="Anomaly", ylabels=False, xlabels=False, 
                          grid=False, fill=False, background='k', linecolor="w", linewidth=1,
                          legend=False, overlap=0.5, figsize=(6,5),kind="counts", bins=80)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
for a in axes[:-1]:
    a.set_xlim([-8,8])      

players = pd.read_csv("data/Players.csv",index_col=0)
players.head()

seasons = pd.read_csv("data/Seasons_Stats.csv", index_col=0)
seasons.head()

joined = seasons.merge(players, on="Player")
threepoints = joined[(joined.Year > 1979) & (joined["FGA"] > 10)].sort_values("Year")
threepoints["3Pfract"] = threepoints["3PA"]/threepoints.FGA

get_ipython().magic('matplotlib inline')

decades = [int(y) if y%10==0 or y == 2017 else None for y in threepoints.Year.unique()]
fig, axes = joypy.joyplot(threepoints, by="Year", column="3Pfract",
                  kind="kde", 
                  range_style='own', tails=0.2, 
                  overlap=3, linewidth=1, colormap=cm.autumn_r,
                  labels=decades, grid='y', figsize=(7,7), 
                  title="Fraction of 3 pointers \n over all field goal attempts")

get_ipython().magic('matplotlib inline')

threepoint_shooters = threepoints[threepoints["3PA"] >= 20] 
decades = [int(y) if y%10==0 or y == 2017 else None for y in threepoint_shooters.Year.unique()]
fig, axes = joypy.joyplot(threepoint_shooters, by="Year", column="3P%",
                   kind="normalized_counts", bins=30, 
                   range_style='all', x_range=[-0.05,0.65],
                   overlap=2, linewidth=1, colormap=cm.autumn_r,
                   labels=decades, grid='both', figsize=(7,7),
                   title="3 Points % \n(at least 20 3P attempts)")

np.random.seed(42)
df = pd.DataFrame(np.random.poisson(10,(24,7)))
df.columns = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df.head()

get_ipython().magic('matplotlib inline')
x_range = list(range(24))
fig, axes = joypy.joyplot(df, kind="values", x_range=x_range)
axes[-1].set_xticks(x_range);

x = np.arange(0,100,0.1)
y =[n*x for n in range(1,4)]

fig, ax = joypy.joyplot(y, labels=["a","b","c"])

labels = ["a","b","c"]
d = {l:v for l,v in zip(labels,y)}
fig, ax = joypy.joyplot(d)

