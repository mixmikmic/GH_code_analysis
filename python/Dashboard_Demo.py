import iris
import param
import numpy as np
import holoviews as hv
import holocube as hc

from cartopy import feature as cf
from paramnb import NbParams, FileSelector

hv.notebook_extension(width=90)
get_ipython().magic("output widgets='live' size=400")
get_ipython().magic('opts Image {+framewise} [colorbar=True] Contours {+framewise}')

class CubeLoader(param.Parameterized):
    
    cube_path = FileSelector(default='files/*.pp')

    cache = {}
    cubes = None
    
    @classmethod
    def load(cls, cube_loader):
        if cls.cube_path not in cls.cache:
            cubelist = iris.load(cls.cube_path)
            for c in cubelist:
                c.coord('grid_longitude').guess_bounds()
                c.coord('grid_latitude').guess_bounds()
            cls.cache[cls.cube_path] = cubelist
        else:
            cubelist = cls.cache[cls.cube_path]
        cubes = {cb.vdims[0].name:cb for cb in [hc.HoloCube(c) for c in cubelist]} # Load cubes into dictionary
        cls.cubes = {k:v for k,v in cubes.items() if k!='unknown'}  # Filter as desired

NbParams(CubeLoader, execute='next', callback=CubeLoader.load)

from cartopy import crs as ccrs
from matplotlib.cm import cmap_d

projections = {k: crs for k, crs in param.concrete_descendents(ccrs.CRS).items()
               if hasattr(crs, '_as_mpl_axes') and not k[0] == '_'}

class CubeBrowser(param.Parameterized):
    """
    CubeBrowser defines a small example GUI to demonstrate
    how to define a small set of widgets to control plotting
    of an iris Cube. It exposes control over the colormap,
    the quantity to be plotted, the element to plot it with
    and the projection to plot it on.
    """

    cmap = param.ObjectSelector(default='viridis',
                                objects=list(cmap_d.keys()))

    quantity = param.ObjectSelector(default=CubeLoader.cubes.keys()[0],
                                    objects=list(CubeLoader.cubes.keys()))

    element = param.ObjectSelector(default=hc.Image,
                                   objects=[hc.Image, hc.Contours])

    projection = param.ObjectSelector(default='default',
                                      objects=['default']+sorted(projections.keys()))
    
    cache = {}

    @classmethod
    def view(cls):
        key = (cls.quantity, cls.element)
        if key in CubeBrowser.cache:
            converted = cls.cache[key]
        else:
            holocube = CubeLoader.cubes[cls.quantity]
            converted = holocube.to(cls.element, ['grid_longitude', 'grid_latitude'], dynamic=True)
            cls.cache[key] = converted
        styled = converted(style={cls.element.name: dict(cmap=cls.cmap)})
        projection = projections[cls.projection]() if cls.projection != 'default' else None
        projected = styled({'Image': dict(plot=dict(projection=projection))}) if projection else styled
        return (projected * hc.GeoFeature(cf.COASTLINE)(plot=dict(scale='50m')))

NbParams(CubeBrowser, execute='next')

# Finally we can declare a cell which uses the settings defined via the widgets to render the requested plot.
# We simply look up the correct cube, convert it to the desired Element type and then display it with the requested options.
CubeBrowser.view()

