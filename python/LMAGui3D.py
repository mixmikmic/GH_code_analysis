get_ipython().magic('matplotlib qt4')
# import matplotlib
# matplotlib.use('nbagg')
#import matplotlib.pyplot as plt
from brawl4d.brawl4d import B4D_startup, redraw
import os

data_path = '/data/tmp/flash_sort_test/'
lma_file = os.path.join(data_path, 'h5_files/2014/May/26/LYLOUT_140526_094000_0600.dat.flash.h5')

get_ipython().magic('pinfo os.path.join')

from datetime import datetime
panels = B4D_startup(basedate=datetime(2014,5,26), ctr_lat=33.5, ctr_lon=-101.5)

import matplotlib.pyplot as plt; plt.show()

from brawl4d.LMA.controller import LMAController
lma_ctrl = LMAController()
d, post_filter_brancher, scatter_ctrl, charge_lasso = lma_ctrl.load_hdf5_to_panels(panels, lma_file)

panels.panels['tz'].axis((9*3600 + 40*60, 9*3600 + 42*60, 1, 15))
panels.panels['xy'].axis((-130, 20, -10, 140))

from brawl4d.LMA.widgets import LMAwidgetController
from IPython.display import display
from brawl4d.LMA.controller import LMAController

lma_tools = LMAwidgetController(panels, lma_ctrl, scatter_ctrl, charge_lasso, d)
display(lma_tools.tools_popup)

chg = d.data['charge']
wh = np.where(chg > 0)
print d.data[wh]['time']

# A reference to the current data in the view is cached by the charge lasso.
current_data = charge_lasso.cache_segment.cache[-1]
# Manually set the color limits on the flash_id variable
scatter_ctrl.default_color_bounds.flash_id =(current_data['flash_id'].min(), current_data['flash_id'].max())
# Color by flash ID.
scatter_ctrl.color_field = 'flash_id'

redraw()

current_events_flashes = lma_ctrl.flash_stats_for_dataset(d, scatter_ctrl.branchpoint)



from scipy.spatial import Delaunay
from scipy.misc import factorial

from stormdrain.pipeline import coroutine
class LMAEventStats(object):
    
    def __init__(self, GeoSys):
        """ GeoSys is an instance of
            stormdrain.support.coords.systems.GeographicSystem instance
        """
        self.GeoSys = GeoSys
    
    def ECEF_coords(self, lon, lat, alt):
        x,y,z = self.GeoSys.toECEF(lon, lat, alt)
        return x,y,z
    
    
    def _hull_volume(self):
        tri = Delaunay(self.xyzt[:,0:3])
        vertices = tri.points[tri.vertices]
        
        # This is the volume formula in 
        # https://github.com/scipy/scipy/blob/master/scipy/spatial/tests/test_qhull.py#L106
        # Except the formula needs to be divided by ndim! to get the volume, cf., 
        # http://en.wikipedia.org/wiki/Simplex#Geometric_properties
        # Credit Pauli Virtanen, Oct 14, 2012, scipy-user list
        q = vertices[:,:-1,:] - vertices[:,-1,None,:]
        simplex_volumes = (1.0 / factorial(q.shape[-1])) * np.fromiter(
                (np.linalg.det(q[k,:,:]) for k in range(tri.nsimplex)) , dtype=float)
        self.tri = tri
        
        # The simplex volumes have negative values since they are oriented 
        # (think surface normal direction for a triangle
        self.volume=np.sum(np.abs(simplex_volumes))
        
    
    @coroutine
    def events_flashes_receiver(self):
        while True:
            evs, fls = (yield)
            x,y,z = self.ECEF_coords(evs['lon'], evs['lat'], evs['alt'])
            t = evs['time']
            self.xyzt = np.vstack((x,y,z,t)).T
            self._hull_volume()
            print "Volume of hull of points in current view is {0:5.1f}".format(
                        self.volume / 1.0e9) # (1000 m)^3
    
    

stats = LMAEventStats(panels.cs.geoProj)
stat_maker = stats.events_flashes_receiver()

current_events_flashes.targets.add(stat_maker)

print current_events_flashes.targets

import scipy.spatial.distance as distance
all_dist_pairs = distance.pdist(stats.xyzt[:,0:2])
sqd=distance.squareform(all_dist_pairs)
sqd.shape

shift_t  = stats.xyzt[:,3]-stats.xyzt[0,3]

fig = plt.figure()
ax=fig.add_subplot(111)
ax.scatter(shift_t, sqd[0,:], cmap='viridis')
t0, t1 = shift_t.min(), shift_t.max()
d0 = 0.0
d_c = 3.0e8*(t1-t0)
d_8 = 1.0e8*(t1-t0)
d_7 = 1.0e7*(t1-t0)
d_6 = 1.0e6*(t1-t0)
d_5 = 1.0e5*(t1-t0)
d_4 = 1.0e4*(t1-t0)
# ax.plot((t0, t1), (d0, d_c), label='c')
# ax.plot((t0, t1), (d0, d_8), label='1e8')
# ax.plot((t0, t1), (d0, d_7), label='1e7')
# ax.plot((t0, t1), (d0, d_6), label='1e6')
ax.plot((t0, t1), (d0, d_5), label='1e5')
ax.plot((t0, t1), (d0, d_4), label='1e4')
ax.legend()

from brawl4d.brawl4d import redraw
import mayavi.mlab as mvlab
from stormdrain.pipeline import coroutine
class MayaviOutlet(object):
    def __init__(self, panels, ev_fl_broadcaster):
        self.ev_fl_broadcaster = ev_fl_broadcaster
        self.ev_fl_broadcaster.targets.add(self.rx())
        self.p3d = mvlab.points3d([0], [0], [0], [0], scale_factor=5e-5)
        self.scene = self.p3d.scene
        self.scene.background = (0,0,0)
        self.panels=panels
        
        # Force a reflow of data
        redraw(panels)
        
        # move camera to see everything after data are plotted
        self.scene.reset_zoom()
        
    @coroutine
    def rx(self):
        while True:
            ev, fl = (yield)
#             self.ev = ev
#             self.fl = fl
            evx, evy, evz, evt = ev['x'], ev['y'], ev['z'], ev['time']
            self.p3d.mlab_source.reset(x=evx, y=evy, z=evz, scalars=evt)
            
current_events_flashes = lma_ctrl.flash_stats_for_dataset(d, scatter_ctrl.branchpoint)
mvo = MayaviOutlet(panels, current_events_flashes)

get_ipython().magic('pinfo mvo.p3d.scene.reset_zoom')

panels.bounds.limits()



