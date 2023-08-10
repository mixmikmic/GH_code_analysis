# we are dependent on numpy, sympy and cvxopt
import numpy as np
import mompy as mp
import cvxopt

import mompy as mp

from IPython.display import display, Markdown, Math, display_markdown
sp.init_printing()
degmm = 2

def printproblem(obj, gs, hs):
    display_markdown('Minimizing', raw=True)
    display(obj)
    display_markdown('subject to', raw=True)
    display(gs)
    

x,y = sp.symbols('x,y')
f = x**2 + 2*y**2
gs = [x + y - 1 <= 1, x >= 0]
hs = None
printproblem(f, gs, hs)

type(gs[0])
a=gs[0]

a.lhs - a.rhs
a.rel_op
type(a)
sp.relational.Equality

MM = mp.MomentMatrix(degmm, (x,y), morder='grevlex')
sol = mp.solvers.solve_GMP(MM, obj, gs, hs, slack = 0)

mp.extractors.extract_solutions_lasserre(MM, sol['x'], 1)

MM.row_monos

x1,x2 = sp.symbols('x1:3')
f = 4*x1**2+x1*x2-4*x2**2-2.1*x1**4+4*x2**4+x1**6/3
gs = []; hs = [];
printproblem(f, gs, hs)
degm = 8

MM = mp.MomentMatrix(degmm, (x1,x2), morder='grevlex')
sol = mp.solvers.solve_GMP(MM, obj, gs, hs, slack = 0)

mp.extractors.extract_solutions_lasserre(MM, sol['x'], 1)

MM.numeric_instance(sol['x'])

MM.row_monos



