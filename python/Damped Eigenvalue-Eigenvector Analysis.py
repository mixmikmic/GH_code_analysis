import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt

# We'll use the scipy version of the linear algebra
from scipy import linalg

# Define the matrices
m1 = 1.0
m2 = 1.0

k1 = 1.0 
k2 = 1.0
k3 = 1.0

M = np.asarray([[m1, 0],
                [0,  m2]])

K = np.asarray([[k1 + k2, -k2],
                [-k2,      k2 + k3]])

eigenvals, eigenvects = linalg.eigh(K, M)

print('\n')
print('The resulting eigenalues are {:.2f} and {:.2f}.'.format(eigenvals[0], eigenvals[1]))
print('\n')
print('So the two natrual frequencies are {:.2f}rad/s and {:.2f}rad/s.'.format(np.sqrt(eigenvals[0]), np.sqrt(eigenvals[1])))
print('\n')

print('\n')
print('The first eigenvector is ' + str(eigenvects[:,0]) + '.')
print('\n')
print('The second eigenvector is ' + str(eigenvects[:,1]) + '.')
print('\n')

# Define a zero damping matrix
c1 = 0.0
c2 = 0.0
c3 = 0.0

C = np.asarray([[c1 + c2, -c2],
                [-c2,      c2 + c3]])



A = np.asarray([[0,                     1,           0,           0],
                [-(k1+k2)/m1, -(c1+c2)/m1,       k2/m1,       c2/m1],
                [0,                     0,           0,           1],
                [k2/m2,             c2/m2, -(k2+k3)/m2, -(c2+c3)/m2]])


eigenvals_ss, eigenvects_ss = linalg.eig(A)

print('\n')
print('The resulting eigenvalues are {:.4}, {:.4}, {:.4}, and {:.4}.'.format(eigenvals_ss[0], eigenvals_ss[1], eigenvals_ss[2], eigenvals_ss[3]))
print('\n')

print('So, the resulting natural frequencies are {:.4}rad/s and {:.4}rad/s.'.format(np.abs(eigenvals_ss[2]), np.abs(eigenvals_ss[0])))
print('\n')

# make 1st entry real
eigvect1_ss = eigenvects_ss[:,0] * np.exp(-1.0j * np.angle(eigenvects_ss[0,0]))
eigvect2_ss = eigenvects_ss[:,2] * np.exp(-1.0j * np.angle(eigenvects_ss[0,2]))

# scale to match the undamped
eigvect1_ss *= (eigenvects[0,0] / eigvect1_ss[0])
eigvect2_ss *= (eigenvects[0,1] / eigvect2_ss[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_ss, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_ss, precision=4, suppress_small=True))
print('\n')

# Form the matrices
A = np.vstack((np.hstack((np.zeros((2,2)),-K)),np.hstack((-K, -C))))

B = np.vstack((np.hstack((-K, np.zeros((2,2)))),np.hstack((np.zeros((2,2)),M))))


# Solve the eigenvalue problem using them
eigenvals_sym, eigenvects_sym = linalg.eig(A, B)

print('\n')
print('The resulting eigenvalues are {:.4}, {:.4}, {:.4}, and {:.4}.'.format(eigenvals_sym[0], eigenvals_sym[1], eigenvals_sym[2], eigenvals_sym[3]))
print('\n')

print('So, the resulting natural frequencies are {:.4}rad/s and {:.4}rad/s.'.format(np.abs(eigenvals_sym[2]), np.abs(eigenvals_sym[0])))
print('\n')

# make 1st entry real
eigvect1_sym = eigenvects_sym[:,0] * np.exp(-1.0j * np.angle(eigenvects_sym[0,0]))
eigvect2_sym = eigenvects_sym[:,2] * np.exp(-1.0j * np.angle(eigenvects_sym[0,2]))

# scale to match the undamped
eigvect1_sym *= (eigenvects[0,0] / eigvect1_sym[0])
eigvect2_sym *= (eigenvects[0,1] / eigvect2_sym[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_sym, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_sym, precision=4, suppress_small=True))
print('\n')

# Define the matrices
m1 = 1.0
m2 = 1.0

k1 = 1.0 
k2 = 1.0
k3 = 1.0

c1 = 0.1
c2 = 0.1
c3 = 0.1

# Redefine the damping matrix
C = np.asarray([[c1 + c2, -c2],
                [-c2,      c2 + c3]])


# Redefine the state-space matrix
A = np.asarray([[0,                     1,           0,           0],
                [-(k1+k2)/m1, -(c1+c2)/m1,       k2/m1,       c2/m1],
                [0,                     0,           0,           1],
                [k2/m2,             c2/m2, -(k2+k3)/m2, -(c2+c3)/m2]])


eigenvals_damped_ss, eigenvects_damped_ss = linalg.eig(A)

print('\n')
print('The resulting eigenvalues are {:.4}, {:.4}, {:.4}, and {:.4}.'.format(eigenvals_damped_ss[0], eigenvals_damped_ss[1], eigenvals_damped_ss[2], eigenvals_damped_ss[3]))
print('\n')

print('So, the resulting natural frequencies are {:.4}rad/s and {:.4}rad/s.'.format(np.abs(eigenvals_damped_ss[2]), np.abs(eigenvals_damped_ss[0])))
print('\n')

# make 1st entry real
eigvect1_damped_ss = eigenvects_damped_ss[:,0] * np.exp(-1.0j * np.angle(eigenvects_damped_ss[0,0]))
eigvect2_damped_ss = eigenvects_damped_ss[:,2] * np.exp(-1.0j * np.angle(eigenvects_damped_ss[0,2]))

# scale to match the undamped
eigvect1_damped_ss *= (eigenvects[0,0] / eigvect1_damped_ss[0])
eigvect2_damped_ss *= (eigenvects[0,1] / eigvect2_damped_ss[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_damped_ss, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_damped_ss, precision=4, suppress_small=True))
print('\n')

# Form the matrices
A = np.vstack((np.hstack((np.zeros((2,2)),-K)),np.hstack((-K, -C))))

B = np.vstack((np.hstack((-K, np.zeros((2,2)))),np.hstack((np.zeros((2,2)),M))))


# Solve the eigenvalue problem using them
eigenvals_damped_sym, eigenvects_damped_sym = linalg.eig(A,B)

# make 1st entry real
eigvect1_damped_sym = eigenvects_damped_sym[:,0] * np.exp(-1.0j * np.angle(eigenvects_damped_sym[0,0]))
eigvect2_damped_sym = eigenvects_damped_sym[:,2] * np.exp(-1.0j * np.angle(eigenvects_damped_sym[0,2]))

# scale to match the undamped
eigvect1_damped_sym *= (eigenvects[0,0] / eigvect1_damped_sym[0])
eigvect2_damped_sym *= (eigenvects[0,1] / eigvect2_damped_sym[0])

print('\n')
print('The first eigevector is ')
print(np.array_str(eigvect1_damped_sym, precision=4, suppress_small=True))
print('\n')
print('The second eigevector is ')
print(np.array_str(eigvect2_damped_sym, precision=4, suppress_small=True))
print('\n')

# Ignore this cell - We just update the CSS to make the notebook look a little bit better and easier to read

# Improve the notebook styling -- Run this first
from IPython.core.display import HTML
css_file = 'styling/CRAWLAB_IPythonNotebook.css'
HTML(open(css_file, "r").read())

