import ginsim

lrg = ginsim.load("http://ginsim.org/sites/default/files/phageLambda4.zginml")

from colomoto.temporal_logics import *

lysogenic = AG(S(CI=2))        # CI is permanently active
lytic = AG(EF(S(CI=0,Cro=2)) & EF(S(CI=0,Cro=3)))  # Cro permanently oscillates between levels 2 and 3
attractors = AG(EF(lysogenic | lytic))   # all the attractors are either lysogenic or lytic
initial_state = S(CI=0,CII=0,Cro=0,N=0)

properties = {
    "s0_lysogenic": If(initial_state, EF(lysogenic)), # lysogenic state is reachable from initial state  
    "s0_lytic": If(initial_state, EF(lytic)),  # lytic state is reachable from initial state
    "attractors": attractors, # all attractors are either lyso or lytic
}

smv_ginsim = ginsim.to_nusmv(lrg)
smv_ginsim.add_ctls(properties)
smv_ginsim.alltrue()

import pypint

an = ginsim.to_pint(lrg)

smv_pint = pypint.to_nusmv(an)
smv_pint.add_ctls(properties)
smv_pint.alltrue()



