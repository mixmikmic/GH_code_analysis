import zipfile, requests, codecs
from io import BytesIO
import pandas as pd
import folium

get_ipython().magic('matplotlib inline')

url = 'http://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_places_national.zip'
r = requests.get(url)
z = zipfile.ZipFile(BytesIO(r.content))
z.extractall()

filename = z.namelist()[0]
filename

file = codecs.open(filename, "r",encoding='utf-8', errors='ignore')
df = pd.read_table(file, sep = '\t')

df = df.rename(columns=lambda x: x.strip())

df.columns

print('There are',len(df),'cities and towns in this database.')

big250 = df[['NAME', 'POP10', 'INTPTLAT', 'INTPTLONG']][df.POP10>250000].sort_values('POP10', ascending=False)

map = folium.Map(location=[30,-97], zoom_start=4,tiles='Stamen Terrain')

marker_cluster = folium.MarkerCluster().add_to(map)

for city in big250.iterrows():
    folium.Marker(
        location = [city[1]['INTPTLAT'],city[1]['INTPTLONG']], 
        popup = city[1]['NAME']+ ' pop:'+str(city[1]['POP10'])
    ).add_to(marker_cluster)
                    
map

