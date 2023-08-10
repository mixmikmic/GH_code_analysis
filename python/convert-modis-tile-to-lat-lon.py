import numpy as np
import urllib2

get_ipython().system('wget https://modis-land.gsfc.nasa.gov/pdf/sn_bound_10deg.txt')

# first seven rows contain header information
# bottom 3 rows are not data
data = np.genfromtxt('sn_bound_10deg.txt', 
                     skip_header = 7, 
                     skip_footer = 3)
lat = 40.015
lon = -105.2705

in_tile = False
i = 0
while(not in_tile):
    in_tile = lat >= data[i, 4] and lat <= data[i, 5] and lon >= data[i, 2] and lon <= data[i, 3]
    i += 1

vert = data[i-1, 0]
horiz = data[i-1, 1]
print('Vertical Tile:', vert, 'Horizontal Tile:', horiz)

