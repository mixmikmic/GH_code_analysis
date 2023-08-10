import ginsim

lrg = ginsim.load("http://ginsim.org/sites/default/files/phageLambda4.zginml")

from colomoto.temporal_logics import *

initial_state = S(CI=0,CII=0,Cro=0,N=0)

lysogenic = AG(S(CI=2))

lytic = AG(S(CI=0) & (S(Cro=2) | S(Cro=3)))

attractors = AG(EF(lysogenic | lytic))

smv = ginsim.to_nusmv(lrg)

smv.add_ctl(attractors, name="global_attractors")

specs = {
    "reach_lyso": If(initial_state, EF(lysogenic)),
    "reach_lytic": If(initial_state, EF(lytic))
}

smv.add_ctls(specs)

smv.verify()

smv.alltrue()

