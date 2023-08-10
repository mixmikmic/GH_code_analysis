get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from flight_safety.queries import get_events_accidents

mpl.rcParams['figure.figsize'] = 10, 6

mpl.rcParams['font.size'] = 20

con = sqlite3.connect('data/avall.db')

events = get_events_accidents(con)

gby_year = events.groupby(events.ev_date.dt.year)

injured_per_year = gby_year[['inj_tot_f', 'inj_tot_s', 'inj_tot_m']].sum()

injured_per_year.plot.area(alpha=0.8)
plt.xlabel('Year')
plt.ylabel('Number of victims');

passengers = pd.read_csv('./data/annual_passengers_carried_data.csv', nrows=1, usecols=range(4,60))
passengers = passengers.transpose()
# renaming column
passengers.columns = ['passengers']
gby_year = events.groupby(events.ev_date.dt.year)

injured_per_year = gby_year[['inj_tot_f', 'inj_tot_s', 'inj_tot_m']].sum()

injured_per_year.tail()
# parsing date in index 
passengers.index = pd.to_datetime(passengers.index.str[:4])
# converting flight number to number
passengers['passengers'] = pd.to_numeric(passengers['passengers'], errors='coerce') / 1e6
passengers.index = passengers.index.year

flights = pd.read_csv('data/API_IS.AIR.DPRT_DS2_en_csv_v2/API_IS.AIR.DPRT_DS2_en_csv_v2.csv', skiprows=4)
flights = flights[flights['Country Name'] == 'United States']

flights = flights.iloc[:, 5:-1].T
flights.index = pd.to_numeric(flights.index)
flights = flights / 1e6

flights.columns = ['flights']

fig, ax = plt.subplots(1, 1)
# ax[0].set_title('Millions of passengers transported')
# passengers['passengers'].plot.area(ax=ax[0], alpha=0.6, color="#0072B2")
# ax[0].set_xlim(1975, 2014)

ax.set_title('Millions of flights')
flights.plot.area(ax=ax, alpha=0.6, legend=False)
ax.set_xlim(1975, 2014)

plt.tight_layout()

accidents_gby_year = events.groupby(events.ev_date.dt.year).ntsb_no.count()
ax = accidents_gby_year.plot.area(alpha=0.8,)
ax.set_xlim(1982, 2015)

ax.set_ylabel('Number of accidents');

accident_rate = accidents_gby_year / flights.flights
accident_rate.tail()

ax = accident_rate.plot.area(alpha=0.8, figsize=(11,5))
ax.set_xlim(1982, 2015)
ax.set_ylabel('accident rate');

ax = accident_rate.plot.area(alpha=0.8, figsize=(11,5))
ax.set_xlim(1998, 2016)
ax.set_ylabel('accident rate');

