from arcgis.gis import GIS
from IPython.display import display
import pandas as pd

groups_df = pd.read_csv('data/groups.csv')

groups_df[:3]

import zipfile
with zipfile.ZipFile("data/Icons.zip") as z:
    z.extractall("data")

import csv

gis = GIS("https://python.playground.esri.com/portal", "arcgis_python", "amazing_arcgis_123")

groups = []
with open('data/groups.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        group = gis.groups.create_from_dict(row)
        groups.append(group)
        

for group in groups:
    display(group)

