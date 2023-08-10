import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import cmocean as cmo
from matplotlib import cm
from scripts import readInput as rInput

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

forcings = rInput.readInput()

forcings.readDEM('data/nodes.csv')

forcings.plotInputGrid(data=forcings.Z,title="elevation",
                       mind=-2000.,maxd=2000,color=cmo.cm.delta)

forcings.readSea('data/sealevel.csv') 

forcings.readRain('data/rain.csv')

forcings.plotInputGrid(data=forcings.rain,title="precipitation [m/y]",
                       mind=0.5,maxd=1,color=cmo.cm.ice_r)

forcings.readDisp('data/disp.csv')

forcings.readFault('data/faults.csv')

forcings.plotInputGrid(data=forcings.disp,title="tectonic [m]",
                       mind=-400.,maxd=400,color=cmo.cm.balance,fault=True)

forcings.readEroLay('data/erolaytop.csv')

forcings.readThickLay('data/erothicktop.csv')

forcings.plotInputGrid(data=forcings.erolays[0],title="erodibility layer 2",
                       mind=1.e-6,maxd=5.e-6,color=cmo.cm.matter)

forcings.plotInputGrid(data=forcings.thicklays[0],title="thickness layer 2",
                       mind=2.,maxd=12,color=cmo.cm.curl)





