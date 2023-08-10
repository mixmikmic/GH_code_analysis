import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# The implementation of the above equations is given below

# Model parameters
H_n = np.array([[ 3],   [2.5],   [2],   [1.5],  [1]])
k_n = np.array([[0.05], [0.1], [0.2], [0.25], [0.4]])

E_0 = np.sum(H_n)

# Initialise the model state parameters
epsilon = 0
chi_n = np.zeros_like(H_n)
alpha_n = np.zeros_like(H_n)

# Define the applied stress history
sigma_max_abs_1 = 0.9
sigma_max_abs_2 = 0

d_sigma_abs = 0.002

sigma_loop = np.append(np.arange(0, sigma_max_abs_1, d_sigma_abs), 
                np.arange(sigma_max_abs_1, -sigma_max_abs_2, -d_sigma_abs))

sigma_history = np.tile(sigma_loop, 10)
epsilon_history = np.zeros(len(sigma_history))

d2_g_d_s2 = -1/E_0
d2_g_d_an2 = -np.matmul(H_n, np.transpose(H_n))/(E_0) + np.diag(H_n[:,0])
d2_g_d_san = -np.transpose(H_n) / E_0
d2_g_d_ans = -H_n / E_0

sigma_0 = 0

# Calculate the incremental response
for index, sigma in enumerate(sigma_history):
        
    d_sigma = sigma - sigma_0
            
    y_n = np.abs(chi_n) - k_n
    d_y_n_d_chi_n = np.sign(chi_n)
    
    # Solve A * lambda_n = b
    b = np.zeros_like(H_n)
    A = np.zeros_like(d2_g_d_an2)
    lambda_n = np.zeros_like(H_n)
    for i_y in range(0,len(H_n)):
        b[i_y,0] = - d_y_n_d_chi_n[i_y] * d2_g_d_ans[i_y] * d_sigma
        A[i_y,:] = d_y_n_d_chi_n[i_y] * d2_g_d_an2[i_y,:] * np.transpose(d_y_n_d_chi_n)
        
    y_active = ((y_n>0) * (d_sigma*d_y_n_d_chi_n>0))[:,0]
    if np.sum(y_active) > 0:
        lambda_active = la.solve(A[y_active,:][:,y_active], b[y_active])
        lambda_n[y_active] = lambda_active
                
    d_alpha_n = lambda_n * d_y_n_d_chi_n
        
    d_epsilon = - (d2_g_d_s2 * d_sigma + np.matmul(d2_g_d_san, d_alpha_n))
    d_chi_n = - (d2_g_d_ans * d_sigma + np.matmul(d2_g_d_an2, d_alpha_n))
            
    epsilon = epsilon + d_epsilon
    chi_n = chi_n + d_chi_n
    alpha_n = alpha_n + d_alpha_n
    
    sigma_0 = sigma
            
    epsilon_history[index] = epsilon   

plt.plot(epsilon_history, sigma_history)
plt.xlabel('$\epsilon$')
plt.ylabel('$\sigma$')



