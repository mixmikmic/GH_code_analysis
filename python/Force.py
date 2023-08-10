from sympy import *
init_printing()
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

r = Symbol('r',positive=True)
V = 1/r
F = - diff(V, r)

F

epos = np.array([[0.0, 1.0, 0.0],
                 [0.2, 0.3, 0.0]])
npos = np.array([[0.0, 0.0, .0]])

def compute_bare_force(npos, epos, F, r):
    forces = np.zeros_like(npos)
    for ion_idx,ion_pos in enumerate(npos):   
        for elec in epos:
            dr = elec - ion_pos
            dr2 = np.dot(dr,dr)
            dr_norm = np.sqrt(dr2)
            #print dr_norm
            dr_hat = dr/dr_norm
            force_component = float(F.subs(r, dr_norm))
            #print dr_norm, 1.0/dr_norm, force_component
            forces[ion_idx] += force_component*dr_hat
            #print ''
    return forces

forces = compute_bare_force(npos, epos, F, r)
print forces

M = Symbol('M', integer=True)
k = Symbol('k', integer=True,positive=True)
a = IndexedBase('a',(M,))
Rc = Symbol('R_c', positive=True)

# Equation 4 in the paper
fbar = Sum(a[k]*r**k, (k,1,M))
fbar

# Integrate individual terms
integrate(r**k,(r,0,Rc)).doit()

# Still need to get to equations 6-8 from equations 2-4

m = Symbol('m')
c = IndexedBase('c',(M,))
j = Symbol('j',integer=True)

Skj = Rc**(m+k+j+1)/(m+k+j+1)
hj = Rc**(j+1)/(j+1)
hj

grfits = OrderedDict()

Mval = 4
mval = 2
S = np.zeros((Mval,Mval))
h = np.zeros(Mval)
Rcval = 0.4
for kval in range(Mval):
    for jval in range(Mval):
        S[kval,jval] = Skj.subs({M:Mval,m:mval,k:kval+1,j:jval+1,Rc:Rcval})
for jval in range(Mval):
    h[jval] = hj.subs({j:jval+1, Rc:Rcval})
print S
print h

ck = np.linalg.solve(S,h)
ck

# Dump in C++ format for inclusion in the unit test
print '// for m_exp=%d, N_basis=%d, Rcut=%f'%(mval, Mval, Rcval)
print 'coeff[%d] = {'%len(ck),
print ','.join(['%g'%ck1 for ck1 in ck]),
#for c in ck:
#    print '%g,'%c,
print '};'

np.dot(np.linalg.inv(S),h)

gr = Sum(c[k]*r**(k+m),(k,1,M))
gr



gr2 = gr.subs({M:Mval, m:mval}).doit()

cc = c.subs(M,Mval)
print 'gr2 = ',gr2
for kval in range(Mval):
    print kval, c[kval+1],ck[kval]
    gr2 = gr2.subs(cc[kval+1],ck[kval])
    print kval,gr2
gr2
grfits[Mval] = gr2

def compute_smoothed_force(npos, epos, F, r):
    forces = np.zeros_like(npos)
    for ion_idx,ion_pos in enumerate(npos):   
        for elec in epos:
            dr = elec - ion_pos
            dr2 = np.dot(dr,dr)
            dr_norm = np.sqrt(dr2)
            #print dr_norm
            dr_hat = dr/dr_norm
            #print 'dr_norm',dr_norm
            if dr_norm < Rcval:
                force_component = float(gr2.subs(r, dr_norm)/dr2)
            else:
                force_component = float(F.subs(r, dr_norm))
            #print dr_norm, 1.0/dr_norm, force_component
            forces[ion_idx] += force_component*dr_hat
            #print ''
    return forces

print 'bare =',compute_bare_force(npos, epos, F, r)
forces = compute_smoothed_force(npos, epos, F, r)
print 'smoothed =',forces

xss = OrderedDict()
yss = OrderedDict()
xs = []
ys = []
step = Rcval/50
for i in range(50):
    rval = i*step + step
    xs.append(rval)
    ys.append(exp(-rval**2))
#xss[0] = xs
#yss[0] = ys
for Mval,gr2 in grfits.iteritems():
    xs = []
    ys = []
    for i in range(50):
        rval = i*step + step
        #print rval, gr2.subs(r,rval)
        xs.append(rval)
        #ys.append(exp(-rval**2)*gr2.subs(r,rval))
        ys.append(gr2.subs(r,rval))
    xss[Mval] = xs
    yss[Mval] = ys


for Mval in xss.keys()[0:1]:
    plt.plot(xss[Mval],yss[Mval])
plt.show()

fx = lambda rval: exp(-rval**2)
print 0,mpmath.quad(fx, [0,Rcval])
for Mval, gr2 in grfits.iteritems():
    #fx = lambda rval : exp(-rval**2)*gr2.subs(r, rval)
    fx = lambda rval : gr2.subs(r,rval)
    ival = mpmath.quad(fx, [0,Rcval])
    #ival = integrate(exp(-r**2)*gr2/r**2,(r,0,Rcval))
    print Mval,ival





