import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import numpy as np
import cmocean as cmo
from matplotlib import cm

from scripts import sectionCarbonate as section

# display plots in SVG format
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')

secmap = section.sectionCarbonate(folder = 'output',step=22)

# Section 1
sec1=np.zeros((2,2))
sec1[0,:] = [287882,469598]
sec1[1,:] = [368169,572448]

# Section 2
sec2=np.zeros((2,2))
sec2[0,:] = [134722,626561]
sec2[1,:] = [257105,500163]

# Section 3
sec3=np.zeros((2,2))
sec3[0,:] = [481670,62921.5]
sec3[1,:] = [603090,26405]

pt = []
pt.append(sec1)
pt.append(sec2)
pt.append(sec3)

secmap.plotSectionMap(title='Elevation', color=cmo.cm.delta, crange=[-5000,5000], pt=pt,size=(8,8))

secCarb1 = section.sectionCarbonate(folder = 'output',step=22)

secCarb1.interpolate(dump=False)
secCarb1.buildSection(sec=sec1)

section.viewSection(width = 700, height = 500, cs = secCarb1, 
            dnlay = 1, rangeX=[30000, 98000], rangeY=[-75,40],
            linesize = 0.5, title='Cross-section 1')

secCarb2 = section.sectionCarbonate(folder = 'output',step=22)

secCarb2.interpolate(dump=False)
secCarb2.buildSection(sec=sec2)

section.viewSection(width = 1000, height = 600, cs = secCarb2, 
            dnlay = 1, rangeX=[0, 170000], rangeY=[-75,120],
            linesize = 0.5, title='Cross-section 2')

secCarb3 = section.sectionCarbonate(folder = 'output',step=22)

secCarb3.interpolate(dump=False)
secCarb3.buildSection(sec=sec3)

section.viewSection(width = 800, height = 500, cs = secCarb3, 
            dnlay = 1, rangeX=[0, 120000], rangeY=[-75,50],
            linesize = 0.5, title='Cross-section 3')



