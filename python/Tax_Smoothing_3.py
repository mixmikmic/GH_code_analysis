import quantecon as qe
import numpy as np
import matplotlib.pyplot as plt
from lq_markov import *
get_ipython().magic('matplotlib inline')

# Model parameters 
beta, Gbar, rho, sigma = 0.95, 5, 0.8, 1

# Basic model matrices
A22 = np.array([[1,0],[Gbar, rho],])
C2 = np.array([[0], [sigma]])
Ug = np.array([[0,1]])

# LQ framework matrices
A_t = np.zeros((1,3))
A_b = np.hstack((np.zeros((2,1)),A22))
A = np.vstack((A_t,A_b))

B = np.zeros((3,1))
B[0,0] = 1

C = np.vstack((np.zeros((1,1)),C2))

Sg = np.hstack((np.zeros((1,1)),Ug))
S1 = np.zeros((1,3))
S1[0,0] = 1
S = S1 + Sg

R = np.dot(S.T,S)

# Large penalty on debt in R2 to prevent borrowing in bad state
R1 = np.copy(R)
R2 = np.copy(R)
R1[0,0] = R[0,0] + 1e-9
R2[0,0] = R[0,0] + 1e12

M = np.array([[-beta]])
Q = np.dot(M.T,M)
W = np.dot(M.T,S)

# Create namedtuple to keep the R,Q,A,B,C,W matrices for each state of the world
world = namedtuple('world', ['A', 'B', 'C', 'R', 'Q', 'W'])

Pi = np.array([[0.95,0,0.05,0],[0.95,0,0.05,0],[0,0.9,0,0.1],[0,0.9,0,0.1]])

#Sets up the four states of the world
v1 = world(A=A,B=B,C=C,R=R1,Q=Q,W=W)
v2 = world(A=A,B=B,C=C,R=R2,Q=Q,W=W)
v3 = world(A=A,B=B,C=C,R=R1,Q=Q,W=W)
v4 = world(A=A,B=B,C=C,R=R2,Q=Q,W=W)

MJLQBarro = LQ_Markov(beta,Pi,v1,v2,v3,v4)

x0 = np.array([[0,1,25]])
T = 300
x,u,w,state = MJLQBarro.compute_sequence(x0,ts_length=T)

# Calculate taxation each period from the budget constraint and the Markov state
tax = np.zeros([T,1])
for i in range(T):
    tax[i,:] = S.dot(x[:,i]) + M.dot(u[:,i])

#Plot of debt issuance and taxation
plt.figure(figsize=(16,4))
plt.subplot(121)
plt.plot(x[0,:])
plt.title('One-period debt issuance')
plt.xlabel('Time')
plt.subplot(122)
plt.plot(tax)
plt.title('Taxation')
plt.xlabel('Time')

M = np.array([[-beta-0.02]])

Q = np.dot(M.T,M)
W = np.dot(M.T,S)

#Sets up the four states of the world
v1 = world(A=A,B=B,C=C,R=R1,Q=Q,W=W)
v2 = world(A=A,B=B,C=C,R=R2,Q=Q,W=W)
v3 = world(A=A,B=B,C=C,R=R1,Q=Q,W=W)
v4 = world(A=A,B=B,C=C,R=R2,Q=Q,W=W)

MJLQBarro2 = LQ_Markov(beta,Pi,v1,v2,v3,v4)
x,u,w,state = MJLQBarro2.compute_sequence(x0,ts_length=T)

# Calculate taxation each period from the budget constraint and the Markov state
tax = np.zeros([T,1])
for i in range(T):
    tax[i,:] = S.dot(x[:,i]) + M.dot(u[:,i])

#Plot of debt issuance and taxation
plt.figure(figsize=(16,4))
plt.subplot(121)
plt.plot(x[0,:])
plt.title('One-period debt issuance')
plt.xlabel('Time')
plt.subplot(122)
plt.plot(tax)
plt.title('Taxation')
plt.xlabel('Time')



