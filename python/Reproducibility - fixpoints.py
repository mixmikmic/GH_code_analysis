import pandas as pd # for displaying list of fixpoints
import ginsim

th17 = ginsim.load("http://ginsim.org/sites/default/files/Th_17.zginml")

import biolqm

th17_lqm = ginsim.to_biolqm(th17)

fps_lqm = biolqm.fixpoints(th17_lqm)
pd.DataFrame(fps_lqm)

import pypint

th17_an = biolqm.to_pint(th17_lqm)

fps_an = pypint.fixpoints(th17_an)
pd.DataFrame(fps_an)

ginsim.show(th17, fps_lqm[1]) # or fps_an[1]



