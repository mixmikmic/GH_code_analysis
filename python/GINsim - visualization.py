import ginsim

lrg = ginsim.load("http://ginsim.org/sites/default/files/SuppMat_Model_Master_Model.zginml")

ginsim.show(lrg)

import biolqm

lqm = ginsim.to_biolqm(lrg)

fps = biolqm.fixpoints(lqm)
print(len(fps), "fixpoints")

ginsim.show(lrg, fps[2])



