import folium

places_on_SAF = [('Point Delgada', 40.0215325, -124.0691973),
('Point Arena', 38.9088, -123.6931),
('Point Reyes', 38.0440, -122.7984),
('Daly City', 37.6879, -122.4702),
('Bautista', 36.8455, -121.5380),
('Parkfield', 35.8997, -120.4327),
('Cholame', 35.7239, -120.2965),
('Bitter Creek National Wildlife Refuge', 34.9342, -119.4005),
('Frazier Park', 34.8228, -118.9448),
('Palmdale', 34.5794, -118.1165),
('San Bernardino', 34.1083, -117.2898),
('Desert Hot Springs', 33.9611, -116.5017),
('Salton Sea State Recreation Area', 33.5088, -115.9181)]

# get the coordinates for these places
lats = [x[1] for x in places_on_SAF]
lons = [x[2] for x in places_on_SAF]
coordinates = zip(lats, lons)

m = folium.Map(location=[36.5,-122], zoom_start=6, tiles='Stamen Terrain')
# Create the map with the appoximate location of San Andreas Fault
SAF=folium.PolyLine(locations=coordinates,weight=5,color = 'red')
m.add_children(SAF)

