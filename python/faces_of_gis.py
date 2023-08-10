import pandas as pd
from arcgis.gis import GIS
from IPython.display import display

df = pd.read_csv('data/lastnames.csv')
df[:3]

gis = GIS()

for index, row in df.iterrows():
    if index == 6:
        break
    users = gis.users.search(row['Surname'])
    for user in users:
        if user['thumbnail'] is not None:
            display(user)

