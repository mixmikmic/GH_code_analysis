from enthought.tvtk.tools import ivtk
from enthought.tvtk.api import tvtk
# Create a cone:
cs = tvtk.ConeSource(resolution=100)
mapper = tvtk.PolyDataMapper(input=cs.output)
actor = tvtk.Actor(mapper=mapper)

# Now create the viewer:
v = ivtk.IVTKWithCrustAndBrowser(size=(600,600))
v.open()
v.scene.add_actors(actor)  # or v.scene.add_actor(a)

