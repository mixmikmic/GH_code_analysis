get_ipython().magic('matplotlib notebook')

import numpy as np
import matplotlib.pyplot as plt

from numpy import array
from dcprogs.likelihood import QMatrix, DeterminantEq, Asymptotes, find_roots, ExactSurvivor
qmatrix = QMatrix( 
           array([[ -3050,        50,  3000,      0,    0 ], 
                  [ 2./3., -1502./3.,     0,    500,    0 ],  
                  [    15,         0, -2065,     50, 2000 ],  
                  [     0,     15000,  4000, -19000,    0 ],  
                  [     0,         0,    10,      0,  -10 ] ]), 2)

from numpy import array
from dcprogs.likelihood import QMatrix, DeterminantEq, Asymptotes, find_roots, ExactSurvivor, eig
qmatrix = QMatrix( 
           array([[ -3050,        50,  3000,      0,    0 ], 
                  [ 2./3., -1502./3.,     0,    500,    0 ],  
                  [    15,         0, -2065,     50, 2000 ],  
                  [     0,     15000,  4000, -19000,    0 ],  
                  [     0,         0,    10,      0,  -10 ] ]), 2)

transitions = qmatrix.transpose()
tau = 1e-4
exact = ExactSurvivor(transitions, tau)
equation = DeterminantEq(transitions, tau)
roots = find_roots(equation)
approx = Asymptotes(equation, roots)
eigenvalues = eig(-transitions.matrix)[0]

def C_i10(i): 
    from numpy import zeros
    result = zeros((transitions.nopen, transitions.nopen), dtype='float64')
    for j in range(transitions.matrix.shape[0]):
        if i == j: continue
        result += np.dot(exact.D_af(i), exact.recursion_af(j, 0, 0)) / (eigenvalues[j] - eigenvalues[i])
        result -= np.dot(exact.D_af(j), exact.recursion_af(i, 0, 0)) / (eigenvalues[i] - eigenvalues[j])
    return result
    
def C_i20(i): 
    from numpy import zeros
    result = zeros((transitions.nopen, transitions.nopen), dtype='float64')
    for j in range(transitions.matrix.shape[0]):
        if i == j: continue
        result += ( np.dot(exact.D_af(i), exact.recursion_af(j, 1, 0)) 
                    + np.dot(exact.D_af(j), exact.recursion_af(i, 1, 0)) ) / (eigenvalues[j] - eigenvalues[i])
        result += ( np.dot(exact.D_af(i), exact.recursion_af(j, 1, 1)) 
                    - np.dot(exact.D_af(j), exact.recursion_af(i, 1, 1)) ) / (eigenvalues[j] - eigenvalues[i])**2
    return result

def C_i21(i): 
    result = np.dot(exact.D_af(i), exact.recursion_af(i, 1, 0)) 
    for j in range(transitions.matrix.shape[0]):
        if i == j: continue
        result -= np.dot(exact.D_af(j), exact.recursion_af(i, 1, 1)) / (eigenvalues[i] - eigenvalues[j])
    return result

def C_i22(i): return np.dot(exact.D_af(i), exact.recursion_af(i, 1, 1)) * 0.5 

print(np.all([np.all(abs(C_i10(i) - exact.recursion_af(i, 1, 0)) < 1e-8) for i in range(5)]))
print(np.all([np.all(abs(C_i20(i) - exact.recursion_af(i, 2, 0)) < 1e-8) for i in range(5)]))
print(np.all([np.all(abs(C_i21(i) - exact.recursion_af(i, 2, 1)) < 1e-8) for i in range(5)]))
print(np.all([np.all(abs(C_i22(i) - exact.recursion_af(i, 2, 2)) < 1e-8) for i in range(5)]))
    

tau, i, j, n = 1e-4, 0, 0, 4



fig = plt.figure()

ax = fig.add_subplot(211)
transitions = qmatrix
exact = ExactSurvivor(transitions, tau)
equation = DeterminantEq(transitions, tau)
roots = find_roots(equation)
approx = Asymptotes(equation, roots)

x = np.arange(0, n * tau, tau / 10.)
ax.plot(x, exact.af(x)[:, i, j], label="exact")
ax.plot(x, approx(x)[:, i, j], '.', label="approx")
ax.set_title("Component ${0}$ of the matrix $^{{A}}R(t)$.".format((i, j)))
ax.legend()


ax = fig.add_subplot(212)
roots = find_roots(equation.transpose())
approx = Asymptotes(equation.transpose(), roots)

i, j = 1, 0
x = np.arange(0, n*tau, tau / 10.)
ax.plot(x, exact.fa(x)[:, i, j], label="exact")
ax.plot(x, approx(x)[:, i, j], '.', label="approx")
ax.set_title("Component ${0}$ of the matrix $^{{F}}R(t)$.".format((i, j)))
ax.legend(loc=0)

fig.tight_layout()



