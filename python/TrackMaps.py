get_ipython().magic('matplotlib inline')
import sys
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import get_cmap

from PlotInterface.maps import MapFigure, saveFigure
import Utilities.shptools as shptools

def make_segments(x, y):
    points = np.array([x,y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

class TrackMapFigure(MapFigure):
    def colorline(self, x, y, z=None, linewidth=1.0, alpha=1.0):
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))

        if not hasattr(z, '__iter__'):
            z = np.array([z])

        z = np.asarray(z)

        segments = make_segments(x, y)
        cmap = ListedColormap(['0.75', '#0FABF6', '#0000FF',
                               '#00FF00', '#FF8100', '#ff0000'])
        norm = BoundaryNorm([0, 17.5, 24.5, 32.5, 44.2, 55.5, 1000], cmap.N)
        lc = LineCollection(segments, array=z, cmap=cmap, 
                            norm=norm, linewidth=linewidth, alpha=alpha)
        
        ax = self.gca()
        ax.add_collection(lc)
        return
    
    def add(self, tracks, xgrid, ygrid, title, map_kwargs):
        self.subfigures.append((tracks, xgrid, ygrid, title, map_kwargs))
        
    def subplot(self, axes, subfigure):
        tracks, xgrid, ygrid, title, map_kwargs = subfigure
        mapobj, mx, my = self.createMap(axes, xgrid, ygrid, map_kwargs)

        for track in tracks:
            mlon, mlat = mapobj(track.Longitude, track.Latitude)
            self.colorline(mlon, mlat, track.WindSpeed, 
                           linewidth=1, alpha=0.75)
        axes.set_title(title)
        self.labelAxes(axes)
        self.addGraticule(axes, mapobj)
        self.addCoastline(mapobj)
        self.fillContinents(mapobj)
        self.addMapScale(mapobj)

class SingleTrackMap(TrackMapFigure):

    def plot(self, tracks, xgrid, ygrid, title, map_kwargs):
        self.add(tracks, xgrid, ygrid, title, map_kwargs)
        super(SingleTrackMap, self).plot()


def saveTrackMap(tracks, xgrid, ygrid, title, map_kwargs, filename):
    fig = SingleTrackMap()
    fig.plot(tracks, xgrid, ygrid, title, map_kwargs)
    saveFigure(fig, filename)



