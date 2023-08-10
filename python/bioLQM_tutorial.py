import pandas as pd # for the visualization of lists of states
import biolqm

lqm = biolqm.load("http://ginsim.org/sites/default/files/boolean_cell_cycle.zginml")

trace = biolqm.trace(lqm)
pd.DataFrame( [s for s in trace] )

trace = biolqm.trace(lqm, "-u synchronous -i 0010000000 -m 50")
pd.DataFrame( [s for s in trace] )

random = biolqm.random(lqm, "-i 0010000000 -m 50")
pd.DataFrame( [s for s in random] )

pd.DataFrame( [s for s in random] )

fps = biolqm.fixpoints(lqm)
pd.DataFrame(fps)

traps = biolqm.trapspace(lqm)
pd.DataFrame(traps)

traps = biolqm.trapspace(lqm, "terminal")
pd.DataFrame(traps)

pert = biolqm.perturbation(lqm, "CycD%1")

fps = biolqm.fixpoints(pert)
pd.DataFrame(fps)

traps = biolqm.trapspace(pert, "terminal")
pd.DataFrame(traps)

pert = biolqm.perturbation(lqm, "CycD%1,Rb%1")

fps = biolqm.fixpoints(pert)
pd.DataFrame(fps)

traps = biolqm.trapspace(pert, "terminal")
pd.DataFrame(traps)

