import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import cmocean as cmo
from matplotlib import cm

from scripts import catchmentErosion as eroCatch

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

catchment = eroCatch.catchmentErosion(folder='output',timestep=13)
catchment.regridTINdataSet()

catchment.plotdataSet(title='Elevation', data=catchment.z, color=cmo.cm.delta,  
                      crange=[-2000,2000], pt=[637936.,4210856.])

catchment.plotdataSet(title='Elevation', data=catchment.z, color=cmo.cm.delta,  
                      crange=[-2000,2000], pt=[637936.,4210856.],
                      erange=[618000,649000,4190000,4230000])

catch = eroCatch.catchmentErosion(folder='output',timestep=13,pointXY=[637936.,4210856.])

catch.regridTINdataSet()

catch.extractRegion(alpha=0.002)

catch.plotCatchment()

catch.getErodedVolume(time=130000.)

catch.plotEroElev(time=130000.)



