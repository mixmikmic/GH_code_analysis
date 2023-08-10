import pandas as pd
df=pd.read_csv("./../data/ntcir12.csv", header=0, sep=",")
df.head(4)

df["location_code"]=df["location"]
df

location_code=pd.read_csv("./../data/locations.csv", sep=";", index_col=0, header=0)
location_code["Log_lat"]=location_code["Log_lat"].astype(str)

l=[]
for index, row in location_code.iterrows():
    x="{"+str(row["locations"])+"','"+str(row["Log_lat"])+"'}"
    l.append(x)

l

len(df['location'].unique())

from geopy.geocoders import Nominatim
# LET'S TRY TO PUT THE LOCATIONS ON A MAP:

geolocator = Nominatim()
locations = df['location'].unique() #list of all locations for user 1

latitude = []
longitude = []
locations=locations[2:]
locations_found=[]

for loc in locations:
    print(loc)
    try:
        location = geolocator.geocode(unicode(loc, "utf-8"))
        latitude.append(location.latitude)
        longitude.append(location.longitude)
        locations_found.append(loc)
    except:
        print('not found')

len(latitude)

latitude_df=pd.DataFrame(latitude)
longitude_df =pd.DataFrame(latitude)
locations=locations[2:]
locations_found=[]

locs=df['location'].unique()
locs=pd.DataFrame(locs)
locs.columns=["locations"]

import plotly
plotly.offline.init_notebook_mode(connected=True)
lons = longitude
lats = latitude

 
data = [ dict(
        type = 'scattergeo',
        locationmode = 'Europe',
        lon = lons,
        lat = lats,
        text = locations_found,
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            )))]

layout = dict(
        title = 'Locations visited by user 1',
        colorbar = True,
        geo = dict(
            scope='europe',
            projection=dict( type='albers us' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
plotly.offline.iplot( fig, validate=False, filename='d3-airports' )

