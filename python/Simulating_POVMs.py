from __future__ import print_function, division
from fractions import Fraction
import numpy as np
import numpy.linalg
import random
import time
from povm_tools import basis, check_ranks, complex_cross_product, dag, decomposePovmToProjective,     enumerate_vertices, find_best_shrinking_factor, get_random_qutrit,     get_visibility, Pauli, truncatedicosahedron

def dp(v):
    result = np.eye(2, dtype=np.complex128)
    for i in range(3):
        result += v[i]*Pauli[i]
    return result

b = [np.array([ 1,  1,  1])/np.sqrt(3),
     np.array([-1, -1,  1])/np.sqrt(3),
     np.array([-1,  1, -1])/np.sqrt(3),
     np.array([ 1, -1, -1])/np.sqrt(3)]
M = [dp(bj)/4 for bj in b]

get_visibility(M)

psi0 = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)], [0]])
omega = np.exp(2*np.pi*1j/3)
D = [[omega**(j*k/2) * sum(np.power(omega, j*m) * np.kron(basis((k+m) % 3), basis(m).T)
                           for m in range(3)) for k in range(1, 4)] for j in range(1, 4)]
psi = [[D[j][k].dot(psi0) for k in range(3)] for j in range(3)]
M = [np.kron(psi[k][j], psi[k][j].conj().T)/3 for k in range(3) for j in range(3)]

get_visibility(M, solver=None, proj=True)

get_visibility(M, solver=None, proj=False)

psi = [get_random_qutrit()]
psi.append(complex_cross_product(psi[0], np.array([[0], [0], [1]])))
psi.append(complex_cross_product(psi[0], psi[1]))
phi = [get_random_qutrit()]
phi.append(complex_cross_product(phi[0], np.array([[0], [0], [1]])))
phi.append(complex_cross_product(phi[0], phi[1]))
M = [0.5*np.kron(psi[0], psi[0].conj().T),
     0.5*np.kron(psi[1], psi[1].conj().T),
     0.5*np.kron(psi[2], psi[2].conj().T) + 0.5*np.kron(phi[0], phi[0].conj().T),
     0.5*np.kron(phi[1], phi[1].conj().T),
     0.5*np.kron(phi[2], phi[2].conj().T),
     np.zeros((3, 3), dtype=np.float64),
     np.zeros((3, 3), dtype=np.float64),
     np.zeros((3, 3), dtype=np.float64),
     np.zeros((3, 3), dtype=np.float64)]

get_visibility(M)

n = 25

# crit is an approximation of 1/(1+sqrt2), the point whose 2D
# stereographic projection is (1/sqrt2, 1/sqrt2)
crit = Fraction(4142, 10000)

# for the interval [crit, 1] the projection from the pole P = (-1, 0)
# approximates "well" the circle
nn = Fraction(1 - crit, n)

# u discretizes the quarter of circle where x, y \geq 0
u = []
for r in range(1, n + 1):
    # P = (0, -1), x \in [crit, 1]
    u.append([Fraction(2*(crit + r*nn), (crit + r*nn)**2 + 1),
              Fraction(2, (crit + r*nn)**2 + 1) - 1])
    # P = (-1, 0), y \in [crit, 1]
    u.append([Fraction(2, (crit + r*nn)**2 + 1) - 1,
              Fraction(2*(crit + r*nn), (crit + r*nn)**2 + 1)])
u = np.array(u)

# u1 discretizes the quarter of circle where x \leq 0, y \geq 0
u1 = np.column_stack((-u[:, 0], u[:, 1]))
u = np.row_stack((u, u1))

# W1 encodes the polyhedron given by the tangency points in u
W1 = np.zeros((u.shape[0] + 1, 9), dtype=fractions.Fraction)
for i in range(u.shape[0]):
    W1[i, 2:5] = np.array([1, -u[i, 0], -u[i, 1]])
# This constraint is to get only the half polygon with positive y2
W1[u.shape[0], 4] = 1

m1 = 2
m2 = 1
# crit is the same as above
mm1 = Fraction(1, m1)
mm2 = Fraction(crit, m2)

# v1 discretizes the positive octant of the sphere
v1 = []

# P = (0, 0, -1), x, y \in [0, 1]
for rx in range(1, m1 + 1):
    for ry in range(1, m1 + 1):
        v1.append([Fraction(2*(rx*mm1), (rx*mm1)**2 + (ry*mm1)**2 + 1),
                   Fraction(2*(ry*mm1), (rx*mm1)**2 + (ry*mm1)**2 + 1),
                   1 - Fraction(2, (rx*mm1)**2 + (ry*mm1)**2 + 1)])

# a second round to improve the approximation around the pole
# P = (0, 0, -1), x, y \in [0, crit]
for rx in range(1, m2 + 1):
    for ry in range(1, m2 + 1):
        v1.append([Fraction(2*(rx*mm2), (rx*mm2)**2 + (ry*mm2)**2 + 1),
                   Fraction(2*(ry*mm2), (rx*mm2)**2 + (ry*mm2)**2 + 1),
                   1 - Fraction(2, (rx*mm2)**2 + (ry*mm2)**2 + 1)])

v1 = np.array(v1)

# we now reflect the positive octant to construct the whole sphere
v1a = np.column_stack((-v1[:, 0], v1[:, 1], v1[:, 2]))
v1 = np.row_stack((v1, v1a))
v1b = np.column_stack((v1[:, 0], -v1[:, 1], v1[:, 2]))
v1 = np.row_stack((v1, v1b))
v1c = np.column_stack((v1[:, 0], v1[:, 1], -v1[:, 2]))
v1 = np.row_stack((v1, v1c))

# the following discretizes the quarters of equators where x, y, z > 0,
# corresponding to the case where rx, ry = 0 above, around the origin
yz = []
xz = []
xy = []
for r in range(1, m1+1):
    # P = [0, 0, -1], x = 0, y \in [0, 1]
    yz.append([0,
               Fraction(2*(r*m1), (r*m1)**2 + 1),
               1 - Fraction(2, (r*m1)**2 + 1)])
    # P = [0, 0,-1], y = 0, x \in [0, 1]
    xz.append([Fraction(2*(r*m1), (r*m1)**2 + 1),
               0,
               1 - Fraction(2, (r*m1)**2 + 1)])
    # P = [0, -1, 0], z = 0, x \in [0, 1]
    xy.append([Fraction(2*(r*m1), (r*m1)**2 + 1),
               1 - Fraction(2, (r*m1)**2 + 1),
               0])

yz = np.array(yz)
xz = np.array(xz)
xy = np.array(xy)

yz1 = np.column_stack((yz[:, 0], -yz[:, 1], yz[:, 2]))
yz2 = np.column_stack((yz[:, 0], yz[:, 1], -yz[:, 2]))
yz3 = np.column_stack((yz[:, 0], -yz[:, 1], -yz[:, 2]))
yz = np.row_stack((yz, yz1, yz2, yz3))

xz1 = np.column_stack((-xz[:, 0], xz[:, 1], xz[:, 2]))
xz2 = np.column_stack((xz[:, 0], xz[:, 1], -xz[:, 2]))
xz3 = np.column_stack((-xz[:, 0], xz[:, 1], -xz[:, 2]))
xz = np.row_stack((xz, xz1, xz2, xz3))

xy1 = np.column_stack((-xy[:, 0], xy[:, 1], xy[:, 2]))
xy2 = np.column_stack((xy[:, 0], -xy[:, 1], xy[:, 2]))
xy3 = np.column_stack((-xy[:, 0], -xy[:, 1], xy[:, 2]))
xy = np.row_stack((xy, xy1, xy2, xy3))

v2 = np.row_stack((yz, xz, xy))

v = np.row_stack((v1, v2))

W2 = np.zeros((v.shape[0], 9), dtype=fractions.Fraction)
for i in range(v.shape[0]):
    W2[i, 5:] = np.array([1, -v[i, 0], -v[i, 1], -v[i, 2]])

W3 = np.zeros((v.shape[0], 9))
for i in range(v.shape[0]):
    W3[i] = [1, -1+v[i, 0], -1, v[i, 0], v[i, 1], -1,
             v[i, 0], v[i, 1], v[i, 2]]

W4 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0]])

W5 = np.array([[1, -1, -1, 0, 0, -1, 0, 0, 0]])

W6 = np.array([[0, 1, -1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, -1, 0, 0, 0],
               [-1, 1, 1, 0, 0, 2, 0, 0, 0]])
hull = np.row_stack((W1, W2, W3, W4, W5, W6))

time0 = time.time()
ext = enumerate_vertices(hull, method="plrs", verbose=1)
print("Vertex enumeration in %d seconds" % (time.time()-time0))

time0 = time.time()
alphas = find_best_shrinking_factor(ext, 2, solver="mosek", parallel=False)
print("\n Found in %d seconds" % (time.time()-time0))

w = np.cos(2*np.pi/3) + 1j*np.sin(2*np.pi/3)
x = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
D = [[], [], []]
for j in range(3):
    for k in range(3):
        D[j].append(np.matrix((w**(j*k/2))*(
            sum(w**(j*m)*x[:, np.mod(k + m, 3)]*dag(x[:, m]) for m in
                range(3)))))

# Discretization of the set of PSD matrices with 9*N elements
N = 2
disc = []
for _ in range(N):
    psi = np.matrix(qutip.Qobj.full(qutip.rand_ket(3)))
    for j in range(3):
        for k in range(3):
            disc.append(D[j][k]*(psi*dag(psi))*dag(D[j][k]))

hull = []
for i in range(9*N):
    # each row of hull ensures tr(M*disc[i])>= 0
    hull.append([np.real(disc[i][2, 2])/3,
                 np.real(disc[i][0, 0]) - np.real(disc[i][2, 2]),
                 2*np.real(disc[i][1, 0]), -2*np.imag(disc[i][1, 0]),
                 2*np.real(disc[i][2, 0]), -2*np.imag(disc[i][2, 0]),
                 np.real(disc[i][1, 1]) - np.real(disc[i][2, 2]),
                 2*np.real(disc[i][2, 1]), -2*np.imag(disc[i][2, 1])])

cov_ext = enumerate_vertices(np.array(hull), method="plrs")

# Converting vectors into covariant POVMs
povms = []
for i in range(cov_ext.shape[0]):
    eff = np.matrix([[cov_ext[i, 1],
                      cov_ext[i, 2] + cov_ext[i, 3]*1j,
                      cov_ext[i, 4] + cov_ext[i, 5]*1j],
                     [cov_ext[i, 2] - cov_ext[i, 3]*1j,
                      cov_ext[i, 6],
                      cov_ext[i, 7] + cov_ext[i, 8]*1j],
                     [cov_ext[i, 4] - cov_ext[i, 5]*1j,
                      cov_ext[i, 7] - cov_ext[i, 8]*1j,
                      1/3 - cov_ext[i, 1] - cov_ext[i, 6]]])
    M = []
    for j in range(3):
        for k in range(3):
            M.append(D[j][k]*eff*dag(D[j][k]))
    povms.append(M)

# Finding the least eigenvalues
A = np.zeros((cov_ext.shape[0]))
for i in range(cov_ext.shape[0]):
        A[i] = min(numpy.linalg.eigvalsh(povms[i][0]))
a = min(A)

alphas = find_best_shrinking_factor(cov_ext, 3, parallel=True)

from qutip import rand_unitary

def get_random_trace_one_povm(dim=3):
    U = rand_unitary(dim)
    M = [U[:, i]*dag(U[:, i]) for i in range(dim)]
    for _ in range(dim-1):
        U = rand_unitary(dim)
        r = random.random()
        for i in range(dim):
            M[i] = r*M[i] + (1-r)*U[:, i]*dag(U[:, i])
    return M

M = get_random_trace_one_povm()
print("Rank of POVM: ", check_ranks(M))
coefficients, projective_measurements = decomposePovmToProjective(M)

print("Ranks of projective measurements: ")
for measurement in projective_measurements:
    print(check_ranks(measurement, tolerance=0.01))

N = coefficients[0]*projective_measurements[0] +     coefficients[1]*(coefficients[2]*projective_measurements[1] + 
    coefficients[3]*(coefficients[4]*(coefficients[6]*projective_measurements[2] + 
    coefficients[7]*projective_measurements[3]) + 
    coefficients[5]*(coefficients[8]*projective_measurements[4] +
                     coefficients[9]*projective_measurements[5])))
not np.any(M - N > 10e-10)

