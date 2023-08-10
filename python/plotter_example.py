from xmitgcm import open_mdsdataset
from plotters import LLC_plotter

# --- Add some libraries for plotting 
import holoviews as hv
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs
hv.notebook_extension()

# Set as appropriate; I've saved only the grid data files in a dedicated
# directory.
data_dir = '/Users/tim/work/results/eccov4r2'
grid_dir = '/Users/tim/work/grids/llc90'
ds = open_mdsdataset(grid_dir=grid_dir, data_dir=data_dir, delta_t=3600, geometry='llc')

# Make a plotter with the dataset as the parent. Note that if you modify the data set
# stored in `ds`, changes *are* stored in the plotter's `parent` variable. However, if
# you rebind `ds`, the plotter still sees the last state of the data set.
plotter = LLC_plotter(ds)

# Have a look at the data set that the plotter stores internally:
plotter.ds

# Now compute some data variable that we want to plot, and
# store it in the data set.
ds['THETA_vol_avg'] = (ds.THETA * ds.rA * ds.drF).sum(dim='k')
plotter.set_data_variable('THETA_vol_avg')

# And now the internal data set looks like:
plotter.ds

# Wrap it into a GeoViews Dataset
kdims = ['time','i','j']
vdims = ['var_to_plot']

gv_ds = gv.Dataset(plotter.ds,kdims=kdims,vdims=vdims,crs=crs.PlateCarree())
print(repr(gv_ds))
print(gv_ds['time'])

get_ipython().run_line_magic('opts', 'Image {+framewise} [colorbar=True]')
get_ipython().run_line_magic('output', "widgets='live'")

gv_ds.to(gv.Image,['j','i']).select(time=2635200)



