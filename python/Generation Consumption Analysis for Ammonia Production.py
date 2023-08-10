from IPython.display import YouTubeVideo
YouTubeVideo("uMkzxV_y7tY",560,315,rel=0)

import sympy

sympy.var('x1 x2 x3')

v = dict()
v['CH4'] = -x1
v['H2O'] = -x1 - x2
v['CO']  = x1 - x2
v['H2']  = 3*x1 + x2 - 3*x3
v['CO2'] = x2
v['N2'] = -x3
v['NH3'] = 2*x3

eqns = [
    sympy.Eq(v['NH3'],1),  
    sympy.Eq(v['CO'],0),
    sympy.Eq(v['H2'],0)
]

soln = sympy.solve(eqns)
print(soln)

for k in v.keys():
    a = v[k].subs(soln)
    if a != 0:
        print("{0:<3s} {1:>6s}".format(k,str(a)))



