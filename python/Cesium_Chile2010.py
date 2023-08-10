import cesiumpy  # install from https://pypi.python.org/pypi/cesiumpy

v = cesiumpy.Viewer()

frameno = 6
kml_file = '../DataFiles/Chile_2010_kml/fig1/frame%sfig1/doc.kml' % str(frameno).zfill(4)
print 'Will plot %s' % kml_file
v.dataSources.add(cesiumpy.KmlDataSource(kml_file))

viewpoint = (-70, -30, 1e7)  # inital view (longitude, latitude, elevation)
v.camera.flyTo(viewpoint)
v

v = cesiumpy.Viewer()

kmz_file = '../DataFiles/Chile_2010_kml/Chile_2010.kmz' 
kml_file2 = '../DataFiles/Chile_2010_kml/fig1/doc.kml' 

v.dataSources.add(cesiumpy.KmlDataSource(kml_file2))

viewpoint = (-70, -30, 1e7)  # inital view (longitude, latitude, elevation)
v.camera.flyTo(viewpoint)
v

v = cesiumpy.Viewer()

kml_file = '../PythonCode/testfig.kml' 

v.dataSources.add(cesiumpy.KmlDataSource(kml_file))

viewpoint = (-123, 47, 1e7)  # inital view (longitude, latitude, elevation)
v.camera.flyTo(viewpoint)
v



