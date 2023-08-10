import sympy
sympy.init_printing()

M, J, Q = sympy.symbols('M, J, Q', positive=True)
t, r = sympy.symbols('t, r', positive=True)
th, ph = sympy.symbols('theta, phi')

coords = t, r, th, ph


Ph = sympy.symbols('Phi', positive=True)
u, v, ut, al = sympy.symbols('u, v, u^t, alpha')
bx, by = sympy.symbols('beta^x, beta^y')

ud, vd = sympy.symbols('u_i, v_i')

vsq = u**2 + v**2
#W = al*ut
W = 1 / sympy.sqrt(1 - ud*u + vd*v)

w = Ph, u, v, ut
U = Ph*W, Ph*W**2*u, Ph*W**2*v, Ph*W**2*al*(ud * (u - bx/al) + vd * (v - by/al))
F_u = Ph*W*(u - bx/al), Ph*W**2*ud*(u - bx/al) + Ph, Ph*W**2*vd*(u - bx/al), Ph*W**2*(u - bx/al)*(ud*bx + vd*by - al)

for x in U:
    print(x.diff(Ph))
    

for x in U:
    print(x.diff(u))

for x in U:
    print(x.diff(v))

for x in U:
    print(x.diff(ut))

for x in F_u:
    print(x.diff(Ph))

for x in F_u:
    print(x.diff(ut))

for x in F_u:
    print(x.diff(u))

for x in F_u:
    print(x.diff(v))

ud = u
bx = 0
W = 1 / sympy.sqrt(1 - ud*u)
dudw = sympy.zeros(3,3)
dudw[0,0] = W
dudw[0,1] = Ph*ud*W**3
dudw[0,2] = Ph*al 

dudw[1,0] = W**2*ud
dudw[1,1] = Ph*W**2*(1 + 2*ud*ud*W**2)
dudw[1,2] = 2*Ph*al*W*ud

dudw[2,0] = W**2*al*ud*(u-bx/al)
dudw[2,1] = Ph*W**2*al*((1 + 2*W**2*ud*ud)*(u-bx/al) + ud)
dudw[2,2] = 2*Ph*al**2*W*ud*(u-bx/al)

dudw

dudw_inv = dudw.inv()
dudw_inv.simplify()

dudw_inv

dfdw = sympy.zeros(3,3)

dfdw[0,0] = W*(u-bx/al)
dfdw[0,1] = Ph*W*(1 + W**2*ud*(u-bx/al))
dfdw[0,2] = Ph*al*(u-bx/al)

dfdw[1,0] = W**2*ud*(u-bx/al) + 1
dfdw[1,1] = Ph*W**2*((1 + 2*W**2*ud*u)*(u-bx/al) + u)
dfdw[1,2] = 2*Ph*al*W*ud *(u-bx/al)

dfdw[2,0] = W**2*(u-bx/al)*(ud*bx - al)
dfdw[2,1] = Ph*W**2*((u-bx/al)*(bx + 2*W**2*u*(ud*bx-al)) + (ud*bx-al))
dfdw[2,2] = 2*Ph*al*W*(u-bx/al)*(ud*bx - al)

dfdw

A = dfdw*dudw_inv

A.simplify()

evals = A.eigenvals()

P, D = A.diagonalize()

print(evals)

for x in evals:
    print(x.simplify())

A

dudw.det()

evals.simplify()

B = sympy.zeros(3,3)
B[0,0] = u
B[0,1] = -1/W
B[0,2] = 1/(al*u*W)
B[1,0] = 2/W
B[1,1] = 2*u - 2/u
B[1,2] = 1/(al*u**2)
B[2,1] = -al
B

w = sympy.symbols('w')
B = B.subs(W,w)

PB, DB = B.diagonalize()

dudw = sympy.zeros(3,3)
dudw[0,0] = w
dudw[0,1] = Ph*ud*w**3
dudw[0,2] = Ph*vd*w**3

dudw[1,0] = w**2*ud
dudw[1,1] = Ph*w**2*(1+2*ud*ud*w**2)
dudw[1,2] = Ph*w**2*2*ud*vd*w**2

dudw[2,0] = w**2*vd
dudw[2,1] = Ph*w**2*2*ud*vd*w**2
dudw[2,2] = Ph*w**2*(1+2*vd**2*w**2)

dudw

dudw.det()

dudw_inv = dudw.inv()

# set bi = 0
dfdw = sympy.zeros(3,3)

dfdw[0,0] = w*(u-bx)
dfdw[0,1] = Ph*w*(1+w**2*ud*(u-bx))
dfdw[0,2] = Ph*w*w**2*vd*(u-bx)

dfdw[1,0] = w**2*ud*(u-bx) + 1
dfdw[1,1] = Ph*w**2*((u-bx)*(1+2*w**2*ud*u)+u)
dfdw[1,2] = Ph*w**2*((u-bx)*(vd*u*w**2))

dfdw[2,0] = w**2*vd*(u-bx)
dfdw[2,1] = Ph*w**2*((u-bx)*(2*w**2*ud*v))
dfdw[2,2] = Ph*w**2*((u-bx)*(1+2*w**2*vd*v)+u)

dfdw

a = dfdw * dudw_inv

a.simplify()

a

evals = a.eigenvals()

for x in evals:
    print(x.simplify())



