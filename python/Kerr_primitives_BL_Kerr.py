get_ipython().magic('display latex')

var('r a', domain='real')

assume(a<1)

assume(a>=0)

f = (r^2 + a^2)/(r^2 - 2*r + a^2)
f

s = integrate(f, r)
s

diff(s, r).simplify_full()

rp = 1 + sqrt(1-a^2)
rm = 1 - sqrt(1-a^2)

F = r + 1/sqrt(1-a^2)*(rp*ln(abs((r-rp)/2)) - rm*ln(abs((r-rm)/2)))
F

dFdr = diff(F,r).simplify_full()
dFdr

bool(dFdr == f)

g = 1/(r^2 - 2*r + a^2)
g

integrate(g,r)

G = 1/(2*sqrt(1-a^2))*ln(abs((r-rp)/(r-rm)))
G

dGdr = diff(G,r).simplify_full()
dGdr

bool(dGdr == g)



