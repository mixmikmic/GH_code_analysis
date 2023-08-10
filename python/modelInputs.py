get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

# Import badlands grid generation toolbox
import pybadlands_companion.toolGeo as simple
# Import badlands displacement map generation toolbox
import pybadlands_companion.toolTec as tec

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

#help(simple.toolGeo.__init__)

dome = simple.toolGeo(extentX=[0.,150000.], extentY=[0.,50000.], dx=500.)

#help(dome.buildDome)

dome.Z = dome.buildDome(a=5000., b=5000., c=1000., base=0., xcenter=75000., ycenter=25000.)

#help(dome.viewGrid)

dome.viewGrid(width=700, height=700, zmin=-800, zmax=2200, zData=dome.Z, title='Export Dome Grid')

#help(dome.buildGrid)

dome.buildGrid(elevation=dome.Z, nameCSV='data/nodes')

dispclass = tec.toolTec(extentX=[0.,150000.], extentY=[0.,50000.], dx=500.)

dispclass.disp = np.zeros(dispclass.Xgrid.shape)
dispclass.disp.fill(1000.)

dispclass.disp[0,:] = 0.
dispclass.disp[-1,:] = 0.
dispclass.disp[:,0] = 0.
dispclass.disp[:,-1] = 0.

dispclass.dispView(width=600, height=600, dispmin=0, dispmax=1000, dispData=dispclass.disp, title='Export Grid')

dispclass.dispGrid(disp=disp, nameCSV='data/displacement')



