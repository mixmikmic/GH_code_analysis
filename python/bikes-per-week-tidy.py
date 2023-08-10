get_ipython().run_line_magic('matplotlib', 'inline')

import os
from urllib.request import urlretrieve

import matplotlib.pyplot as plt

import pandas as pd

def get_velo_data(location):
    URL = "https://data.stadt-zuerich.ch/dataset/verkehrszaehlungen_werte_fussgaenger_velo/resource/ed354dde-c0f9-43b3-b05b-08c5f4c3f65a/download/2016verkehrszaehlungenwertefussgaengervelo.csv"

    if not os.path.exists("2016.csv"):
        urlretrieve(URL, "2016.csv")
        
    data = pd.read_csv("2016.csv", parse_dates=True, index_col='Datum')

    # filter by location
    data = data[data.Standort == location]

    # subselect only the Velo data
    data = data[["Velo_in", "Velo_out"]]
    
    data['Total'] = data.Velo_in + data.Velo_out
    
    return data

mythenquai = get_velo_data('ECO09113499')

# rename for easier plotting
mythenquai.columns = ["North", "South", "Total"]

daily = mythenquai.resample('W').sum()
hourly = mythenquai.resample('H').sum()

daily.plot()
plt.legend(loc='best');

hourly.groupby(hourly.index.time).mean().plot()



