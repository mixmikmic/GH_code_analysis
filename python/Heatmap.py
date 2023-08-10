import os
import gmaps
import data_jam.models as m

# Best City In The World
NYC = (40.7128, -74.0059)

# If only Google Maps didn't requrie a key...
gmaps.configure(api_key=os.environ['GOOGLE_MAPS_API_KEY'])

# Grab all the requests coordinates from the DB
query = m.ServiceRequest.select().limit(1000000).order_by(m.ServiceRequest.id.desc())
all_requests = m.ServiceRequest.lat_lngs(query)

# Create a map of NYC
map_fig = gmaps.figure(center=NYC, zoom_level=11)

# Generate a heatmap
heatmap = gmaps.heatmap_layer(all_requests)
heatmap.max_intensity = 1000
map_fig.add_layer(heatmap)

map_fig

from datetime import date
import tabulate
from IPython.display import HTML, display
import data_jam.models as m

start = date(2012, 10, 29)
end = date(2012, 11, 30)
storms = m.Storm.select().where(m.Storm.date >= start, m.Storm.date <= end)
headers = ('Date', 'Type', 'Borough', 'Deaths', 'Injured')
table = []

for storm in storms.iterator():
    table.append((storm.date, storm.type, storm.borough, storm.deaths, storm.injured))
    
display(HTML(tabulate.tabulate(table, headers=headers, tablefmt='html')))

import pendulum
import data_jam.models as m

start = pendulum.create(2012, 10, 22, 0, 0, 0, 0, 'America/New_York')
end = pendulum.create(2012, 11, 8, 23, 59, 59, 0, 'America/New_York')

query = m.ServiceRequest.select().where(
    m.ServiceRequest.happened_between(start, end)
)
calls = query.count()
print(f"{calls} calls were made.")

import pendulum
import matplotlib.pyplot as plt
import data_jam.models as m

from IPython.core.pylabtools import figsize
figsize(24, 7)

start = pendulum.create(2011, 10, 22, 0, 0, 0, 0, 'America/New_York')
end = pendulum.create(2013, 11, 8, 23, 59, 59, 0, 'America/New_York')
calls = list(m.ServiceRequest.count_by_day(start, end))

plt.plot(
    [call[0] for call in calls], 
    [call[1] for call in calls],
)
plt.xticks(rotation=90)
plt.show()



