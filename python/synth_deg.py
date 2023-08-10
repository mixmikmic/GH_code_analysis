from pysb import *

Model()

Monomer('A')

Parameter('kA_syn', 1e0)
Parameter('kA_deg', 1e-1)
Rule('synthesize_A', None >> A(), kA_syn)
Rule('degrade_A', A() >> None, kA_deg)

Parameter('A_0', 1.0)
Initial(A(), A_0)

Observable('A_total', A())



