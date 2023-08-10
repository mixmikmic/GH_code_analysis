import numpy as np
import pandas as pd

from bokeh.io import show, output_notebook
from bokeh.plotting import figure

output_notebook()

flights = pd.read_csv('data/flights.csv', index_col=0)
df = flights.copy()

flights = flights[['arr_delay', 'origin', 'dest', 'air_time', 'distance', 'carrier']]
flights['air_speed (mph)'] = flights['distance'] / (flights['air_time'] / 60)
flights.head()

flights_grouped = flights.groupby(['origin', 'dest', 'carrier']).agg(['count', 'mean', 'min', 'max']).reset_index()
flights_grouped.head()

airlines = pd.read_csv('data/airlines.csv')
airlines_dict = {code: name for code, name in zip(airlines['carrier'], airlines['name'])}
flights_grouped['carrier'] = flights_grouped['carrier'].replace(airlines_dict)
flights_grouped.head()

set(airports['TZ'])

from bokeh.sampledata.airport_routes import airports
airports = airports[airports['IATA'].isin(list(set(flights_grouped['origin'])) + list(set(flights_grouped['dest'])))]
airports = airports[~airports['TZ'].isin(['America/Anchorage', 'Pacific/Honolulu'])]
airports.head()

airports_long = {code: longitude for code, longitude in zip(airports['IATA'], airports['Longitude'])}
airports_lati = {code: latitude for code, latitude in zip(airports['IATA'], airports['Latitude'])}

flights_grouped['start_long'] = flights_grouped['origin'].replace(airports_long)
flights_grouped['start_lati'] = flights_grouped['origin'].replace(airports_lati)
flights_grouped['end_long'] = flights_grouped['dest'].replace(airports_long)
flights_grouped['end_lati'] = flights_grouped['dest'].replace(airports_lati)
flights_grouped = flights_grouped[~flights_grouped['end_long'].isin(['ANC','BQN', 'HNL', 'PSE', 'SJU', 'STT'])]
flights_grouped.head()

from bokeh.sampledata.us_states import data as states

if 'HI' in states: del states['HI']
if 'AK' in states: del states['AK']

xs = [states[state]['lons'] for state in states]
ys = [states[state]['lats'] for state in states]

p = figure(plot_width=1200, plot_height=760, title = 'United States')
p.xaxis.visible=False
p.yaxis.visible=False
p.grid.visible=False
p.patches(xs, ys, fill_alpha = 0.0, line_color = 'gray')

show(p)

x = [[start_long, end_long] for start_long, end_long in zip(flights_grouped['start_long'], flights_grouped['end_long'])]
y = [[start_lati, end_lati] for start_lati, end_lati in zip(flights_grouped['start_lati'], flights_grouped['end_lati'])]

p.multi_line(x, y, color = 'red')
show(p)

flights_grouped.to_csv('data/flights_map.csv')



