import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action = "ignore", category = FutureWarning)

import glob
import lavavu
from scripts import viewBadlands as visu

get_ipython().run_line_magic('matplotlib', 'inline')

folder = 'output'

stepCounter = len(glob.glob1(folder+"/xmf/","tin.time*"))-1
print "Number of visualisation steps created: ",stepCounter

tin,flow,sea = visu.loadStep(folder,stepCounter)

visu.view1Step(tin, flow, sea, scaleZ=10, maxZ=3000, maxED=20, maxwED=10, flowlines=False)

visu.viewTime(folder, steps=stepCounter, it=1, scaleZ=10, maxZ=3000, maxED=20, maxwED=10, flowlines=True)









