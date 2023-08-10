from __future__ import division
from sympy import *
x, y, z, t = symbols('x y z t')
f, g, h = symbols('f g h', cls=Function)
init_printing()

L = symbols('L')
a1, a2, a3, b1, b2, b3, c1, c2, c3 = symbols('a1 a2 a3 b1 b2 b3 c1 c2 c3')
u0, ux, uy, uz, v0, vx, vy, vz, w0, wx, wy, wz = symbols('u0 u_x u_y u_z                                                         v0 v_x v_y v_z                                                         w0 w_x w_y w_z')
lamda, mu, rho = symbols('lamda mu rho')

u = u0 + ux*sin(a1*pi*x/L) + uy*sin(a2*pi*y/L) + uz*sin(a3*pi*z/L)
v = v0 + vx*sin(b1*pi*x/L) + vy*sin(b2*pi*y/L) + vz*sin(b3*pi*z/L)
w = w0 + wx*sin(c1*pi*x/L) + wy*sin(c2*pi*y/L) + wz*sin(c3*pi*z/L)

def laplacian(U, X=[x, y, z]):
    lap_U = Matrix([sum([diff(U[k], X[j], 2) for j in range(3)]) for k in range(3)])
    return lap_U  


def div(U, X=[x, y, z]):
    return sum([diff(U[k], X[k]) for k in range(3)])


def grad(f, X=[x, y, z]):
    return Matrix([diff(f, X[k]) for k in range(3)])


def rot(U, X=[x, y, z]):
    return Matrix([diff(U[2], X[1]) - diff(U[1], X[2]),
                   diff(U[0], X[2]) - diff(U[2], X[0]),
                   diff(U[1], X[0]) - diff(U[0], X[1])])

def navier(U, X=[x,y,z], lamda=lamda, mu=mu):
    return mu*laplacian(U, X) + (lamda + mu)*grad(div(U, X), X)


def dt(U, t=t):
    return Matrix([diff(U[k], t) for k in range(3)])

U = [u,v,w]
F = rho*dt(U) - navier(U)

simplify(F)

L = symbols('L')
h0, hx, hy = symbols('h0 hx hy')
lamda, mu, rho = symbols('lamda mu rho')
alpha, mu_c, xi, gamma, J = symbols('alpha mu_c xi gamma J')

u = u0 + ux*sin(a1*pi*x/L) + uy*sin(a2*pi*y/L)
v = v0 + vx*sin(b1*pi*x/L) + vy*sin(b2*pi*y/L)
w = 0
h1 = h0 + hx*sin(c1*pi*x/L) + hy*sin(c2*pi*y/L)
h2 = 0
h3 = 0

def coss(U, H, X=[x,y,z], lamda=lamda, mu=mu, mu_c=mu_c, xi=xi, gamma=gamma):
    f_u = (lamda + 2*mu)*grad(div(U)) - (mu + mu_c)*rot(rot(U)) + 2*mu_c*rot(H)
    f_h = (alpha + 2*gamma + 2*xi)*grad(div(H)) - 2*xi*rot(rot(H)) + 2*mu_c*rot(U) - 4*mu_c*Matrix(H)
    return f_u, f_h

U = [u, v, w]
H = [h1, h2, h3]
f_u, f_h = coss(U, H)

f_u

f_h





from IPython.core.display import HTML
def css_styling():
    styles = open('./styles/custom_barba.css', 'r').read()
    return HTML(styles)
css_styling()



