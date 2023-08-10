# The implementation of the above equations is given below
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Model parameters
E = 10
H_n = np.array([  8,   6,   4,   2,  1])
k_n = np.array([0.2, 0.4, 0.6, 0.8,  1])
N = len(H_n)

# Initialise the model state parameters
epsilon = 0
chi_n = np.zeros(N)
alpha_n = np.zeros(N)

# Define the applied stress history
sigma_max_abs_1 = 0.9
sigma_max_abs_2 = 0.9
sigma_max_abs_3 = 0.7

d_sigma_abs = 0.01
sigma_history = np.append(np.append(np.arange(0, sigma_max_abs_1, 
        d_sigma_abs), np.arange(sigma_max_abs_1, -sigma_max_abs_2, 
        -d_sigma_abs)), np.arange(-sigma_max_abs_2, sigma_max_abs_3, d_sigma_abs))
epsilon_history = np.zeros(len(sigma_history))

d2_g_d_s2 = -1/E
d2_g_d_a2 =  H_n
d2_g_d_sa = -np.ones(N)
d2_g_d_as = -np.ones(N)

sigma_0 = 0

# Calculate the incremental response
for index, sigma in enumerate(sigma_history):
    
    d_sigma = sigma - sigma_0
        
    y_n = np.abs(chi_n) - k_n
    d_y_n_d_chi_n = np.sign(chi_n)
        
    lambda_n = -(d_y_n_d_chi_n * d2_g_d_sa)/(d_y_n_d_chi_n * d2_g_d_a2 * d_y_n_d_chi_n) * d_sigma 
    lambda_n[lambda_n < 0] = 0
    lambda_n[y_n < 0] = 0
                
    d_alpha_n = lambda_n * d_y_n_d_chi_n
        
    d_epsilon = - (d2_g_d_s2 * d_sigma + np.sum(d2_g_d_sa * d_alpha_n))
    d_chi_n = - (d2_g_d_as * d_sigma + d2_g_d_a2 * d_alpha_n)
        
    epsilon = epsilon + d_epsilon
    chi_n = chi_n + d_chi_n
    alpha_n = alpha_n + d_alpha_n
    
    sigma_0 = sigma
            
    epsilon_history[index] = epsilon   

plt.plot(epsilon_history, sigma_history)
plt.xlabel('$\epsilon$')
plt.ylabel('$\sigma$')

