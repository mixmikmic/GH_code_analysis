import pygplates
import healpy as hp
import numpy as np


### Make a multipoint feature
### with points evenly distributed points on the sphere
nSide = 32
othetas,ophis = hp.pix2ang(nSide,np.arange(12*nSide**2))
othetas = np.pi/2-othetas
ophis[ophis>np.pi] -= np.pi*2

lats = np.degrees(othetas) 
lons = np.degrees(ophis)

# The next line exlicitly creates the feature as a 'MeshNode' gpml type, so that
# GPlates will display velocities at each point
multipoint_feature = pygplates.Feature(
    pygplates.FeatureType.create_from_qualified_string('gpml:MeshNode'))
multipoint = pygplates.MultiPointOnSphere(zip(lats,lons))  
multipoint_feature.set_geometry(multipoint)
multipoint_feature.set_name('Equal Area points from healpy')

output_feature_collection = pygplates.FeatureCollection(multipoint_feature)
    
output_feature_collection.write('healpix_mesh_feature.gpmlz')


# N is simply the total number of points that we want to create
N = 20000

## Marsaglia's method
dim = 3

norm = np.random.normal
normal_deviates = norm(size=(dim, N))

radius = np.sqrt((normal_deviates**2).sum(axis=0))
points = normal_deviates/radius

# The above code returns points on a sphere, but specified in 3D cartesian
# space rather than lat/long space. However, we can use these directly to 
# create the multipoint feature, pygplates will recognise them as x,y,z

multipoint_feature = pygplates.Feature()
multipoint_feature.set_geometry(pygplates.MultiPointOnSphere((points.T)))
multipoint_feature.set_name('Random Points from Marsaglia''s method')

multipoint_feature_collection = pygplates.FeatureCollection(multipoint_feature)

multipoint_feature_collection.write('pseudo_random_points_feature.gpmlz')



