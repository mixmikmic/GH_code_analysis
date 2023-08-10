import grass.script as grass

get_ipython().system('g.gisenv')

get_ipython().system('gdalinfo /usr/local/share/data/north_carolina/rast_geotiff/basin_50K.tif')

proj4 = '+proj=lcc +lat_1=36.1667 +lat_2=34.333 +lat_0=33.75 +lon_0=-79 +x_0=609601.22 +y_0=0 +no_defs +a=6378137 +rf=298.2572221010042 +to_meter=1'
#this will return an error if the location alreay exist (just ignore it) <-- fix me
try:
    grass.run_command('g.proj', proj4=proj4, location='nc')
except:
    print 'grass location nc already exist'

get_ipython().system('g.mapset location=nc mapset=PERMANENT')

get_ipython().system('g.proj -p')

