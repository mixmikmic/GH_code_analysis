import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def macaulay(x):
    x[x<0] = 0
    return x

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

mu = 0.01
dt = 0.01

# Calculate the incremental response
for index, sigma in enumerate(sigma_history):
        
    d_sigma = sigma - sigma_0
            
    d_w_d_chi_n = 1 / mu * macaulay(np.abs(chi_n) - k_n) * np.sign(chi_n)
                    
    d_alpha_n = d_w_d_chi_n * dt
        
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

