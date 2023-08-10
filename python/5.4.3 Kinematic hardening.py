# The implementation of the above equations is given below
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Model parameters
E = 1
H = 0.3
k = 1

# Initialise the model state parameters
epsilon = 0
chi = 0
alpha = 0

# Define the applied stress history
sigma_max_abs_1 = 1.5
sigma_max_abs_2 = 1.5
sigma_max_abs_3 = 0.8

d_sigma_abs = 0.01
sigma_history = np.append(np.append(np.arange(0, sigma_max_abs_1, 
        d_sigma_abs), np.arange(sigma_max_abs_1, -sigma_max_abs_2, 
        -d_sigma_abs)), np.arange(-sigma_max_abs_2, sigma_max_abs_3, d_sigma_abs))
epsilon_history = np.zeros(len(sigma_history))

d2_g_d_s2 = -1/E
d2_g_d_a2 =  H
d2_g_d_sa = -1
d2_g_d_as = -1

sigma_0 = 0

# Calculate the incremental response
for index, sigma in enumerate(sigma_history):
    
    d_sigma = sigma - sigma_0
    
    y = np.abs(chi) - k
    d_y_d_chi = 2*chi
    
    if y > 0 and d_sigma * chi > 0:
        lambda_ = (d_y_d_chi * d2_g_d_sa)/(- d_y_d_chi * d2_g_d_a2 * d_y_d_chi) * d_sigma
    else:
        lambda_ = 0
        
    d_alpha = lambda_ * d_y_d_chi
    
    d_epsilon = - (d2_g_d_s2 * d_sigma + d2_g_d_sa * d_alpha)
    d_chi = - (d2_g_d_as * d_sigma + d2_g_d_a2 * d_alpha)
    
    epsilon = epsilon + d_epsilon
    chi = chi + d_chi
    alpha = alpha + d_alpha
    
    sigma_0 = sigma
        
    epsilon_history[index] = epsilon    

plt.plot(epsilon_history, sigma_history)
plt.xlabel('$\epsilon$')
plt.ylabel('$\sigma$')

