from colomoto_jupyter import tabulate
import biolqm

lqm = biolqm.load("/tmp/colomoto5xd4zsjj_colomotoa6ug09_yphageLambda4.zginml")

fps = biolqm.fixpoints(lqm)
tabulate(fps)



