# Import libraries
import os
import folium
import pandas as pd
from ast import literal_eval

# Color of the Paris Metro lines
# Got it from https://fr.wikipedia.org/wiki/Mod%C3%A8le:M%C3%A9tro_de_Paris/couleur_fond
color_table = pd.read_table('./data/color_metro_paris.txt', sep=',', header=None)

# Files containing the position of the Paris metro stations and lines 
# This one can be downloaded from https://opendata.stif.info/explore/dataset/emplacement-des-gares-idf-data-generalisee/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true
filename_gares = './data/emplacement-des-gares-idf.csv'
# This one can be downloaded from https://opendata.stif.info/explore/dataset/emplacement-des-gares-idf/download/?format=csv&timezone=Europe/Berlin&use_labels_for_header=true
filename_lignes = './data/traces-du-reseau-ferre-idf.csv'

# output map filename
outfp = "paris_metro.html"

# Paris
m = folium.Map(location=[48.859553, 2.336332], 
               zoom_start=12.5, 
               control_scale=True, 
               prefer_canvas=True)

# Add tiles
folium.TileLayer('OpenStreetMap').add_to(m)
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('Stamen Watercolor').add_to(m)
folium.TileLayer('CartoDB dark_matter').add_to(m)
folium.TileLayer('CartoDB positron').add_to(m)

# We create a feature group (layer) for each set line
feature_group1 = folium.FeatureGroup(name='Line 1')
feature_group2 = folium.FeatureGroup(name='Line 2')
feature_group3 = folium.FeatureGroup(name='Line 3')
feature_group4 = folium.FeatureGroup(name='Line 3b')
feature_group5 = folium.FeatureGroup(name='Line 4')
feature_group6 = folium.FeatureGroup(name='Line 5')
feature_group7 = folium.FeatureGroup(name='Line 6')
feature_group8 = folium.FeatureGroup(name='Line 7')
feature_group9 = folium.FeatureGroup(name='Line 7b')
feature_group10 = folium.FeatureGroup(name='Line 8')
feature_group11 = folium.FeatureGroup(name='Line 9')
feature_group12 = folium.FeatureGroup(name='Line 10')
feature_group13 = folium.FeatureGroup(name='Line 11')
feature_group14 = folium.FeatureGroup(name='Line 12')
feature_group15 = folium.FeatureGroup(name='Line 13')
feature_group16 = folium.FeatureGroup(name='Line 14')
feature_group17 = folium.FeatureGroup(name='RER A')
feature_group18 = folium.FeatureGroup(name='RER B')
feature_group19 = folium.FeatureGroup(name='RER C')
feature_group20 = folium.FeatureGroup(name='RER D')
feature_group21 = folium.FeatureGroup(name='RER E')

# And a dictionary associating a line string to a group
dct = {'1': feature_group1, 
       '2': feature_group2, 
       '3': feature_group3,
       '3b': feature_group4,
       '4': feature_group5,
       '5': feature_group6,
       '6': feature_group7,
       '7': feature_group8,
       '7b': feature_group9,
       '8': feature_group10,
       '9': feature_group11,
       '10': feature_group12,
       '11': feature_group13,
       '12': feature_group14,
       '13': feature_group15,
       '14': feature_group16,
       'A': feature_group17,
       'B': feature_group18,
       'C': feature_group19,
       'D': feature_group20,
       'E': feature_group21
      }

# Get the lines
df = pd.read_table(filename_lignes, sep=';')
# only the metro and RER ones
df_M = df.loc[(df['METRO']==1) | (df['RER']==1)]

for index, row in df_M.iterrows():
    # which line it is
    line_str = row['INDICE_LIG']
    # get the right color
    color_metro_line = color_table.iat[color_table.loc[color_table[0]==line_str].index.values[0],1]
    
    # get the lines data and format it properly
    data = literal_eval(row['Geo Shape'])
    points = []
    for point in data['coordinates']:
        points.append(tuple([point[1], point[0]]))
    
    # add to the correct layer
    folium.PolyLine(points, 
                    color=color_metro_line,
                    weight=2,
                    opacity=1).add_to(dct[line_str])


# Get the stations
df = pd.read_table(filename_gares, sep=';')
# only the metro and RER ones
df_M = df.loc[(df['METRO']==1) | (df['RER']==1)]

for index, row in df_M.iterrows():
    # which lines pass through
    line_str = row['INDICE_LIG']
    
    # Get the station name
    popup_text = '{}'.format(row['NOM_GARE'].replace("'","\\'"))
    
    # get the right color
    color_metro_line = color_table.iat[color_table.loc[color_table[0]==line_str].index.values[0],1]
    color_metro_line_fill = color_metro_line
    # plotting radius
    rad = 3

    # add to the correct layer
    folium.CircleMarker(literal_eval(row['Geo Point']),
                        color=color_metro_line,
                        radius=rad,
                        fill=True,
                        fill_color=color_metro_line_fill,
                        fill_opacity=1,
                        popup=popup_text,
                        ).add_to(dct[line_str])

# add the layers to the map
feature_group1.add_to(m)
feature_group2.add_to(m)
feature_group3.add_to(m)
feature_group4.add_to(m)
feature_group5.add_to(m)
feature_group6.add_to(m)
feature_group7.add_to(m)
feature_group8.add_to(m)
feature_group9.add_to(m)
feature_group10.add_to(m)
feature_group11.add_to(m)
feature_group12.add_to(m)
feature_group13.add_to(m)
feature_group14.add_to(m)
feature_group15.add_to(m)
feature_group16.add_to(m)
feature_group17.add_to(m)
feature_group18.add_to(m)
feature_group19.add_to(m)
feature_group20.add_to(m)
feature_group21.add_to(m)

# enable layer control
folium.LayerControl().add_to(m)

m.save(outfp)

m



