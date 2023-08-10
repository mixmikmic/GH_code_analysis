URL = "https://data.stadt-zuerich.ch/dataset/verkehrszaehlungen_werte_fussgaenger_velo/resource/ed354dde-c0f9-43b3-b05b-08c5f4c3f65a/download/2016verkehrszaehlungenwertefussgaengervelo.csv"

from urllib.request import urlretrieve

urlretrieve(URL, "2016.csv")

get_ipython().system('head 2016.csv')

import pandas as pd

data = pd.read_csv("2016.csv")

data.head()

data = pd.read_csv("2016.csv", parse_dates=True, index_col='Datum')

data.head()

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 4
plt.rcParams["lines.markersize"] = 10
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'

data.Velo_in.plot()

data.Standort.value_counts()

location = 'ECO09113499'
mythenquai = data[data.Standort == location]

mythenquai.head()

mythenquai.Velo_in.plot()
mythenquai.Velo_out.plot()
plt.legend(loc='best')

daily = mythenquai.resample('W').sum()

daily.Velo_in.plot()
daily.Velo_out.plot()
plt.legend(loc='best')

# subselect only the Velo data
mythenquai = mythenquai[["Velo_in", "Velo_out"]]
mythenquai = mythenquai.resample("H").sum()

mythenquai.head()

# rename for easier plotting
mythenquai.columns = ["North", "South"]

mythenquai['Total'] = mythenquai.North + mythenquai.South

mythenquai.groupby(mythenquai.index.time).mean().plot()

pivoted = mythenquai.pivot_table("Total", index=mythenquai.index.time,
                                 columns=mythenquai.index.date)

pivoted.iloc[:5,:5]

pivoted.plot(legend=False, alpha=0.05, color='k', rot=30);

week = mythenquai[mythenquai.index.dayofweek.isin([0,1,2,3,4])]
week.pivot_table("Total", index=week.index.time,
                              columns=week.index.date).plot(legend=False, alpha=0.1,
                                                            color='k', rot=30)

weekend = mythenquai[mythenquai.index.dayofweek.isin([5,6])]
weekend.pivot_table("Total",
                    index=weekend.index.time,
                    columns=weekend.index.date).plot(legend=False, alpha=0.1, 
                                                     color='k', rot=30)

for day in range(7):
    daily = mythenquai[mythenquai.index.dayofweek == day]
    daily.pivot_table("Total", index=daily.index.time,
                                  columns=daily.index.date).plot(legend=False, alpha=0.1,
                                                                color='k', rot=30)
    plt.show()

