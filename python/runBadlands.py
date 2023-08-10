get_ipython().run_line_magic('matplotlib', 'inline')

# Import badlands grid generation toolbox
import pybadlands_companion.toolGeo as simple

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# help(simple.toolGeo.__init__)
wave = simple.toolGeo(extentX=[0.,30000.], extentY=[0.,30000.], dx=100.)

# help(wave.buildWave)
wave.Z = wave.buildWave(A=-400., P=30000., base=300., xcenter=15000.)

# It is possible to have a look at the grid surface 
# using **plotly** library before proceeding to the creation of the badlands surface.
# 
# help(tecwave.dispView)
wave.viewGrid(width=600, height=600, zmin=-500, zmax=500, zData=wave.Z, title='Export Grid')

# help(wave.buildGrid)
wave.buildGrid(elevation=wave.Z, nameCSV='data/node')

get_ipython().run_line_magic('matplotlib', 'inline')

# Import badlands grid generation toolbox
import pybadlands_companion.toolSea as tools

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

sea = tools.toolSea()
sea.buildCurve(timeExt = [0.,1000000.], timeStep = 10000., seaExt = [0.,0.], 
                   ampExt = [20.,20.], periodExt = [500000.,500000.])
# Visualize
sea.plotCurves(fsize=(4,5), figName = 'Sea level')

# Export the sea-level file  to the data folder
sea.exportCurve(nameCSV='data/sealevel_1myr')

from pyBadlands.model import Model as badlandsModel

# initialise model
model = badlandsModel()
# Define the XmL input file
model.load_xml('basin.xml')

model.run_to_time(1000000)





