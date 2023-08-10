import matplotlib.pyplot as plt
from math import sqrt
from sympy import integrate, N
from ncpol2sdpa import SdpRelaxation, generate_variables, flatten
get_ipython().magic('matplotlib inline')

x = generate_variables('x')[0]
sdp = SdpRelaxation([x])
sdp.get_relaxation(1, objective=x**2)
sdp.solve()
print(sdp.primal, sdp.dual)

moments = [x-1, 1-x]
sdp = SdpRelaxation([x])
sdp.get_relaxation(1, objective=x**2, momentinequalities=moments)
sdp.solve()
print(sdp.primal, sdp.dual)

coeffs = [-sdp.extract_dual_value(0, range(1))]
coeffs += [sdp.y_mat[2*i+1][0][0] - sdp.y_mat[2*i+2][0][0]
           for i in range(len(moments)//2)]
sigma_i = sdp.get_sos_decomposition()
print(coeffs, [sdp.dual, sigma_i[1]-sigma_i[2]])

def J(x):
    return -(1-x**2)*x


def Jk(x, coeffs):
    return sum(ci*x**i for i, ci in enumerate(coeffs))

x = generate_variables('x')[0]
y = generate_variables('y')[0]

level = 4
gamma = [integrate(x**i, (x, 0, 1)) for i in range(1, 2*level+1)]
marginals = flatten([[x**i-N(gamma[i-1]), N(gamma[i-1])-x**i] for i in range(1, 2*level+1)])

f = -x*y**2
inequalities = [1.0-x**2-y**2, 1-x, x, 1-y, y]

sdp = SdpRelaxation([x, y], verbose=0)
sdp.get_relaxation(level, objective=f, momentinequalities=marginals,
                   inequalities=inequalities)
sdp.solve()
print(sdp.primal, sdp.dual, sdp.status)
coeffs = [sdp.extract_dual_value(0, range(len(inequalities)+1))]
coeffs += [sdp.y_mat[len(inequalities)+1+2*i][0][0] - sdp.y_mat[len(inequalities)+1+2*i+1][0][0]
           for i in range(len(marginals)//2)]

x_domain = [i/100. for i in range(100)]
plt.plot(x_domain, [J(xi) for xi in x_domain], linewidth=2.5)
plt.plot(x_domain, [Jk(xi, coeffs) for xi in x_domain], linewidth=2.5)
plt.show()

def J(x):
    return -sqrt(x**2+(1-x)**2)

x = generate_variables('x')[0]
y = generate_variables('y', 2)

f = x*y[0] + (1-x)*y[1]

gamma = [integrate(x**i, (x, 0, 1)) for i in range(1, 2*level+1)]
marginals = flatten([[x**i-N(gamma[i-1]), N(gamma[i-1])-x**i] for i in range(1, 2*level+1)])
inequalities = [1-y[0]**2-y[1]**2, x, 1-x]
sdp = SdpRelaxation(flatten([x, y]))
sdp.get_relaxation(level, objective=f, momentinequalities=marginals,
                   inequalities=inequalities)
sdp.solve()
print(sdp.primal, sdp.dual, sdp.status)
coeffs = [sdp.extract_dual_value(0, range(len(inequalities)+1))]
coeffs += [sdp.y_mat[len(inequalities)+1+2*i][0][0] - sdp.y_mat[len(inequalities)+1+2*i+1][0][0]
           for i in range(len(marginals)//2)]
plt.plot(x_domain, [J(xi) for xi in x_domain], linewidth=2.5)
plt.plot(x_domain, [Jk(xi, coeffs) for xi in x_domain], linewidth=2.5)
plt.show()

def J(x):
    return -2*abs(1-2*x)*sqrt(x/(1+x))

x = generate_variables('x')[0]
y = generate_variables('y', 2)
f = (1-2*x)*(y[0] + y[1])

gamma = [integrate(x**i, (x, 0, 1)) for i in range(1, 2*level+1)]
marginals = flatten([[x**i-N(gamma[i-1]), N(gamma[i-1])-x**i] for i in range(1, 2*level+1)])

inequalities = [x*y[0]**2 + y[1]**2 - x,  - x*y[0]**2 - y[1]**2 + x,
                y[0]**2 + x*y[1]**2 - x,  - y[0]**2 - x*y[1]**2 + x,
                1-x, x]
sdp = SdpRelaxation(flatten([x, y]))
sdp.get_relaxation(level, objective=f, momentinequalities=marginals,
                   inequalities=inequalities)
sdp.solve(solver="mosek")
print(sdp.primal, sdp.dual, sdp.status)
coeffs = [sdp.extract_dual_value(0, range(len(inequalities)+1))]
coeffs += [sdp.y_mat[len(inequalities)+1+2*i][0][0] - sdp.y_mat[len(inequalities)+1+2*i+1][0][0]
           for i in range(len(marginals)//2)]
plt.plot(x_domain, [J(xi) for xi in x_domain], linewidth=2.5)
plt.plot(x_domain, [Jk(xi, coeffs) for xi in x_domain], linewidth=2.5)
plt.show()

x = generate_variables('x')[0]
y = generate_variables('y')[0]

f = x + y
g = [x <= 1.0, x >= -1.0]
G = x*y**2/2.0 - y**3/3.0
h = [y <= 1.0, y >= -1.0]
epsilon = 0.001
M = 1.0
level = 3

def lower_level(k, G, h, M):
    gamma = [integrate(x**i, (x, -M, M))/(2*M) for i in range(1, 2*k+1)]
    marginals = flatten([[x**i-N(gamma[i-1]), N(gamma[i-1])-x**i] for i in range(1, 2*k+1)])
    inequalities = h + [x**2 <= M**2]
    lowRelaxation = SdpRelaxation([x, y])
    lowRelaxation.get_relaxation(k, objective=G,
                                 momentinequalities=marginals,
                                 inequalities=inequalities)
    lowRelaxation.solve()
    print("Low-level:", lowRelaxation.primal, lowRelaxation.dual, lowRelaxation.status)
    coeffs = []
    for i in range(len(marginals)//2):
        coeffs.append(lowRelaxation.y_mat[len(inequalities)+1+2*i][0][0] -
                      lowRelaxation.y_mat[len(inequalities)+1+2*i+1][0][0])
    blocks = [i for i in range(len(inequalities)+1)]
    constant = lowRelaxation.extract_dual_value(0, blocks)
    return constant + sum(ci*x**(i+1) for i, ci in enumerate(coeffs))

Jk = lower_level(level, G, h, M)
inequalities = g + h + [G - Jk <= epsilon]
highRelaxation = SdpRelaxation([x, y], verbose=0)
highRelaxation.get_relaxation(level, objective=f,
                              inequalities=inequalities)
highRelaxation.solve()
print("High-level:", highRelaxation.primal, highRelaxation.status)
print("Optimal x and y:", highRelaxation[x], highRelaxation[y])

