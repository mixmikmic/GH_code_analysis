get_ipython().magic('run grassutil.ipy')

DATADIR='/home/user/data/north_carolina/rast_geotiff/'

get_ipython().system('g.mapset location=nc mapset=PERMANENT')

get_ipython().system('g.proj -p')

get_ipython().system('r.in.gdal input={DATADIR}/elevation.tif output=elevation -e --o')

makeImage(basemap='elevation', inputlayer='elevation', maptype='raster', 
          vsize=10, maptitle='elevation', region=region2dict(rast='elevation'), legend=False, outputimagename='test.png')

get_ipython().system('r.in.gdal input={DATADIR}/basin_50K.tif output=basin_50K -e --o')

makeImage(basemap='basin_50K', inputlayer='basin_50K', maptype='raster', 
          vsize=10, maptitle='Basins', region=region2dict(rast='basin_50K'), legend=False, outputimagename='test.png')

get_ipython().system('r.blend first=elevation second=basin_50K output=basin_relief percent=30 --overwrite --quiet')

makeImage(basemap='basin_50K', inputlayer='basin_relief', maptype='rgb', 
          vsize=10, maptitle='Basins', region=region2dict(rast='basin_50K'), legend=False, outputimagename='test.png')

get_ipython().system('m.nviz.image elevation_map=elevation output=elevation position=0.5,0.5               perspective=100 height=800 color_map=basin_50K               resolution_fine=1 resolution_coarse=1 format=tif --q')

get_ipython().system('convert elevation.tif elevation.png')
Image("elevation.png")



