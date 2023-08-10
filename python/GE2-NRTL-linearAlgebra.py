# Ethyl acetate (1) + water (2) + ethanol (3)


alpha12 = 0.4

alpha23 = 0.3

alpha13 = 0.3

# 6 binary Aij parameters
Dg12 = 1335 * 4.184 #J/K
Dg21 = 2510 * 4.184 #J/K

Dg23 = 976 * 4.184 #J/K
Dg32 = 88 * 4.184 #J/K

Dg13 = 301 * 4.184 #J/K
Dg31 = 322 * 4.184 #J/K

import numpy as np
from scipy.constants import R
print(R)

#assemble matrix with regressed parameters Dg_i,j, according to the model all diagonal terms are zero
Dg = np.array([[0, Dg12, Dg13],
             [Dg21, 0, Dg23],
             [Dg31, Dg32, 0]])

A = Dg/R

#assemble symmetric matrix alpha
alpha = np.array([[0, alpha12, alpha13],
                [alpha12, 0, alpha23],
                [alpha13, alpha23, 0]])

def Gamma(T,x,alpha,A):
    tau=np.zeros([3,3])
    for j in range(3):
        for i in range(3):
            tau[j,i]=A[j,i]/T    
    
    G=np.zeros([3,3])
    for j in range(3):
        for i in range(3):
            G[j,i]=np.exp((-alpha[j,i]*tau[j,i]))
    
    Gamma=np.zeros([3])
    for i in range(3):

        Sj1=0
        Sj2=0
        Sj3=0
        for j in range(3):
            Sj1 += tau[j,i]*G[j,i]*x[j]
            Sj2 += G[j,i]*x[j]
    
            Sk1=0
            Sk2=0
            Sk3=0
            for k in range(3):
                Sk1 += G[k,j]*x[k]
                Sk2 += x[k]*tau[k,j]*G[k,j]
                Sk3 += G[k,j]*x[k]
            
            Sj3 += ((x[j]*G[i,j])/(Sk1))*(tau[i,j]-(Sk2)/(Sk3))
        
        Gamma[i]=np.exp(Sj1/Sj2 + Sj3)
    
    return Gamma

#test it to see if results match
#trial temperature and composition:
T = 293.15 #K
x=np.array([.1,.3,.6]) #normalized
ans=Gamma(T,x,alpha,A)
print(ans) #ttest using those trial input

def Gamma_linalg(T,c_x,q_alpha, q_A): # here we chose to use the starting letters s, c, l, and Q to identify scalar variables, single column matrixes, single line matrixes and square matrixes to the reader
    # e_T should be an scalar value for temperature
    # c_x should be a single-column matrix(2d array) representing composition
    # q_alpha should be two matrix(2d array) representing component dependent parameters inferred from experimental data
    # q_tau should be two matrix(2d array) representing component dependent parameters inferred from experimental data
    
    q_tau     = q_A/T #element-wise division by scalar
    q_at      = q_alpha*q_tau #M2d * N2d yields element-wise multiplication
    q_minusat     = -q_at #element wise signal change
    q_G       = np.exp(q_minusat) #element wise exponentiation
    q_Lambda  = (q_tau*q_G) #element-wise multiplication
    q_GT      = q_G.T #M.T yields the transpose matrix of M;
    c_den     = q_GT @ c_x #M @ N yields the matrix multiplication between M and N
    c_invden  = 1/c_den #scalar 1 is broadcast for element-wise division
    l_invden  = c_invden.T #transposition of a single column matrix yields a single line matrix
    q_E       = q_Lambda * l_invden #element wise multiplication between (nl,nc) matrix M with (1,nc) matrix l broadcasts the element-wise multiplication of each (1,nc) submatrix of M with the unique (1,nc) matrix l
    q_L       = q_G * l_invden #broadcasting of element-wise multiplication
    l_x       = c_x.T #transposition of a single column matrix yields a single line matrix
    q_Lx      = q_L * l_x #broadcasting of element-wise multiplication
    q_ET      = q_E.T #transposition of square matrix
    q_LxET    = q_Lx @ q_ET #matrix multiplication
    q_ES      = q_E+q_ET #element-wise sum
    q_ESminusLxET = q_ES-q_LxET #element-wise subtraction
    q_ESminusLxETx     = q_ESminusLxET @ c_x #matrix multiplication
    gamma     = np.exp(q_ESminusLxETx) #element-wise exponentiation
    return gamma

#a test case for the function
#where x was the composition represented in a 1d array
#and now line and x_as_column is a single lne and a single column matrix, respectively to represent composition
#We build it using the array function to wrap the 1d array in another 1d aray, hence a 2d array
x_as_line = np.array([x])
#We transpose x_as_line to creata x_as_column, which is the shape expected by the linalgGamma function
x_as_column = np.array([x]).T #we wrap x with an extra braket so it is now a 2d array, a matrix, as we did not add any extra lines it is a single-line matrix, we tranpose to generate a single-column matrix (1d arrays cannot be transposed, there is no second dimension)
#print the output to see if errors occur and if values are coherent(between zero and infinity, tending to 1 for ideal solutions)
print(Gamma_linalg(T,x_as_column,alpha,A)) #test using those trial input

def Gamma_linalg_tiny(T,c_x,q_alpha, q_A):
    #note that we used many lines for didatics
    #we can do it in few lines:
    #note that some expression occur more than once below
    #so it may be useful define it as a intermediary recurrent term here
    #and calculate it once to use it then several times
    q_tau     = q_A/T
    q_G       = np.exp(-(q_alpha*q_tau))
    l_D       = ((1/((q_G.T) @ c_x)).T)
    q_E       = (q_tau*q_G) * l_D 
    gamma     = np.exp(((q_E+(q_E.T))-(((q_G * l_D) * (c_x.T)) @ (q_E.T))) @ c_x)
    return gamma

#test it to see that the results are the same
print(Gamma_linalg_tiny(T,x_as_column,alpha,A)) #test using those trial input

get_ipython().magic('timeit Gamma(T,x,alpha,A) #ttest using those trial input #My result was 90 micro seconds per loop')

get_ipython().magic('timeit Gamma_linalg(T,x_as_column,alpha,A) #ttest using those trial input #My result was 25 micro seconds per loop')

get_ipython().magic('timeit Gamma_linalg_tiny(T,x_as_column,alpha,A) #ttest using those trial input #My result was 25 micro seconds per loop')

get_ipython().run_cell_magic('timeit', '', '#approximately time the random number generation to subtract later\n# ~21 micro seconds per loop here\nN=3\nx=np.random.rand(N,1)\nx=x/sum(x)\nalpha=np.random.rand(N,N)\nA=np.random.rand(N,N)\nT=(np.random.rand(1,1)+.5)*273')

get_ipython().run_cell_magic('timeit', '', '# ~440 micro seconds per loop here (420 subtracting the random number generation)\nN=3\nx=np.random.rand(N,1)\nx=x/sum(x)\nalpha=np.random.rand(N,N)\nA=np.random.rand(N,N)\nT=(np.random.rand(1,1)+.5)*273\n\n_=Gamma(\n    T,\n    x,\n    alpha,\n    A)')

get_ipython().run_cell_magic('timeit', '', '# ~56 micro seconds per loop here  (36 subtracting the random number generation)\nN=3\nx=np.random.rand(N,1)\nx=x/sum(x)\nalpha=np.random.rand(N,N)\nA=np.random.rand(N,N)\nT=(np.random.rand(1,1)+.5)*273\n\n_=Gamma_linalg(\n    T,\n    x,\n    alpha,\n    A)')

get_ipython().run_cell_magic('timeit', '', '# ~52 micro seconds per loop here (32 subtracting the random number generation)\nN=3\nx=np.random.rand(N,1)\nx=x/sum(x)\nalpha=np.random.rand(N,N)\nA=np.random.rand(N,N)\nT=(np.random.rand(1,1)+.5)*273\n\n_=Gamma_linalg_tiny(\n    T,\n    x,\n    alpha,\n    A)')

#These two lines is all that it takes to accelerate this function
from numba import jit
@jit
#now repeat the function with a different bname so we can compare
def Gamma_numba(T,x,alpha,A):

    tau=np.zeros([3,3])
    for j in range(3):
        for i in range(3):
            tau[j,i]=A[j,i]/T    
    
    G=np.zeros([3,3])
    for j in range(3):
        for i in range(3):
            G[j,i]=np.exp((-alpha[j,i]*tau[j,i]))
    
    Gamma=np.zeros([3])
    for i in range(3):

        Sj1=0
        Sj2=0
        Sj3=0
        for j in range(3):
            Sj1 += tau[j,i]*G[j,i]*x[j]
            Sj2 += G[j,i]*x[j]
    
            Sk1=0
            Sk2=0
            Sk3=0
            for k in range(3):
                Sk1 += G[k,j]*x[k]
                Sk2 += x[k]*tau[k,j]*G[k,j]
                Sk3 += G[k,j]*x[k]
            
            Sj3 += ((x[j]*G[i,j])/(Sk1))*(tau[i,j]-(Sk2)/(Sk3))
        
        Gamma[i]=np.exp(Sj1/Sj2 + Sj3)
    
    return Gamma

from numba import jit
@jit
def lngammaNRTL(T,c_x,q_alpha, q_A):
    q_tau     = q_A/T
    q_G       = np.exp(-(q_alpha*q_tau))
    l_D       = ((1/((q_G.T)@
                c_x)).T)
    q_E       = (q_tau*q_G)*l_D 
    return (((q_E+(q_E.T))-(((q_G*l_D)*(c_x.T))@
            (q_E.T)))@
            c_x)

#These two lines are all that it takes to accelerate this function
from numba import jit
@jit
#now repeat the function with a different name so we can compare them
def Gamma_linalg_tiny_numba(T,c_x,q_alpha, q_A):
    q_tau     = q_A/T
    q_G       = np.exp(-(q_alpha*q_tau))
    l_D       = ((1/((q_G.T) @ c_x)).T)
    q_E       = (q_tau*q_G) * l_D 
    gamma     = np.exp(((q_E+(q_E.T))-(((q_G * l_D) * (c_x.T)) @ (q_E.T))) @ c_x)
    return gamma

get_ipython().run_cell_magic('timeit', '', '# ~370 micro seconds per loop here (350 subtracting the random number generation, versus 420 thats not much acceleration)\nN=3\nx=np.random.rand(N,1)\nx=x/sum(x)\nalpha=np.random.rand(N,N)\nA=np.random.rand(N,N)\nT=(np.random.rand(1,1)+.5)*273\n\n_ = Gamma_numba(\n    T,\n    x,\n    alpha,\n    A)')

get_ipython().run_cell_magic('timeit', '', '# ~34 micro seconds per loop here (14 subtracting the random number generation, versus 32 thats approximately half)\n\nN=3\nx=np.random.rand(N,1)\nx=x/sum(x)\nalpha=np.random.rand(N,N)\nA=np.random.rand(N,N)\nT=(np.random.rand(1,1)+.5)*273\n\n_ = Gamma_linalg_tiny_numba(\n    T,\n    x,\n    alpha,\n    A)')





