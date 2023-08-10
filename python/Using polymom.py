from __future__ import print_function
import sympy as sp
import numpy as np
import BorderBasis as BB
np.set_printoptions(precision=3)
from IPython.display import display, Markdown, Math
sp.init_printing()

R, x, y = sp.ring('x,y', sp.RR, order=sp.grevlex)

I = [ x**2 + y**2 - 1.0, x + y ]

B = BB.BorderBasisFactory(1e-5).generate(R,I)

print("=== Generator Basis:")
for f in B.generator_basis:
    display(f.as_expr())
print("=== Quotient Basis:")
for f in B.quotient_basis:
    display(f.as_expr())
print("=== Variety:")
for v in B.zeros():
    print(zip(R.symbols, v))

import globals # Hacky way I'm keeping track of intermediate information.
for stage, iteration, W in globals.info.Ws:
    display(Markdown("### Expansion %d, Iteration %d"%(stage, iteration)))
    for w in W:
        display(globals.info.L.as_poly(w).as_expr())

