import pypint

m = pypint.load("http://ginsim.org/sites/default/files/drosophilaCellCycleVariants.zginml")

m2 = pypint.load("https://cellcollective.org/#2329/apoptosis-network")

m3 = pypint.load()

n = m.having(CycE=1)      # having can take keyword arguments
n = m.having({"Notch": 1, "Stg": 1})   # .. or a dictionnary
n = m.having("NoCycD_Endocycle")       # .. or the name of a registered state

from pypint import Goal # avoid typing pypint
simple = Goal("g=1")    # simple goal
substate = Goal("a=1,b=1")   # reach a state where both a=1 and b=1
seq = Goal("a=1,b=1", "c=1") # reach a state where a=1 and b=1 and from c=1 is reachable
alt = Goal("a=1", "b=1") | Goal("c=1") # either reach a state where a=1 and then a state where b=1, 
                                       # or reach a state where c=1

invasion = pypint.load("http://ginsim.org/sites/default/files/SuppMat_Model_Master_Model.zginml")

invasion.having(DNAdamage=1,ECMicroenv=1).reachability("Apoptosis=1")

goal = Goal("Apoptosis=1") | Goal("CellCycleArrest=1")
invasion.initial_state["ECMicroenv"] = 1
invasion.initial_state["DNAdamage"] = 0
mutations = invasion.oneshot_mutations_for_cut(goal, maxsize=3, exclude={"ECMicroenv","DNAdamage"})
mutations

invasion.lock({'DKK1': 1, 'NICD': 0}).reachability(goal)

invasion.cutsets("Apoptosis=1",maxsize=3)

invasion.disable(p53=1).reachability("Apoptosis=1")

invasion.bifurcations("Apoptosis=1")

len(invasion.local_transitions)

red = invasion.reduce_for_goal("Apoptosis=1")
len(red.local_transitions)

th17 = pypint.load("http://ginsim.org/sites/default/files/Th_17.zginml")

fps = th17.fixpoints()
import pandas as pd # for pretty display of fixpoints
pd.DataFrame(fps)

phage = pypint.load("http://ginsim.org/sites/default/files/phageLambda4.zginml",simplify=False)

phage.initial_state

attractors = phage.reachable_attractors()
attractors

phage.having(attractors[0]["sample"]).reachable_stategraph() # display the cyclic attractor

