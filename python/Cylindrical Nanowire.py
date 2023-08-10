import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import itertools
import scipy.special
import copy

# functional creation of the Hamiltonian

def calc_H(physics):
    N_z = physics.N_z
    N_phi = physics.N_phi
    t_z = physics.t_z
    t_phi = physics.t_phi
    Delta = physics.Delta
    mu = physics.mu
    flux = physics.flux
    
    def calc_H_element(e1,e2):
        (z1,phi1) = e1
        (z2,phi2) = e2
        # onsite element
        if z1 == z2 and phi1 == phi2:
            diag_ele = 2*t_z + np.abs(t_phi)*(2 + (2*np.pi*flux/N_phi)**2) - mu
            return np.array([[diag_ele,Delta],[np.conj(Delta),-np.conj(diag_ele)]])
        # z hopping
        elif abs(z1-z2) == 1 and phi1 == phi2:
            return np.array([[-t_z,0],[0,t_z]])
        # phi hopping
        elif (phi1-phi2 == 1 or phi1-phi2 == N_phi-1)and z1 == z2:
            return np.array([[-t_phi,0],[0,np.conj(t_phi)]])
        elif (phi1-phi2 == -1 or phi1-phi2 == -N_phi+1)and z1 == z2:
            return np.conj(np.array([[-t_phi,0],[0,np.conj(t_phi)]])).T
        else:
            return np.array([[0,0],[0,0]])
        # the basis is given by (n_z,n_phi) where n_z = 0,..,N_z-1, n_phi = 0,...,N_phi-1
    basis = list(itertools.product(range(N_z),range(N_phi)))
    H = [calc_H_element(e1,e2) for e1 in basis for e2 in basis]
    N = (N_phi*N_z)

    H_ar = np.array(H,dtype=np.complex64).reshape((N,N,2,2))
    H_mat = np.array([H_ar[x,:,y,:].flatten() for x in range(H_ar.shape[0]) for y in range(H_ar.shape[2])])    .flatten().reshape(2*N,2*N)

    return H_mat
    
    
    
    

def current_E(E,physics):
   N_phi = physics.N_phi
   N_z =  physics.N_z
   flux = physics.flux
   
   eta = physics.eta
   kT = physics.kT
   mu_1 = physics.mu_1
   mu_2 = physics.mu_2
   mu = 0.5*(mu_1 + mu_2)
   Delta1 = physics.Delta1
   Delta2 = physics.Delta2
   
   t_z = physics.t_z
   t_phi = physics.t_phi
   
   # create the physical paramters dictionaries to create the respective Hamiltonians for S1-N-S2
   physical_parameters_N = Physics()
   physical_parameters_N.N_z = N_z
   physical_parameters_N.N_phi = N_phi
   physical_parameters_N.Delta = 0.0
   physical_parameters_N.t_z = t_z 
   physical_parameters_N.t_phi = t_phi 
   physical_parameters_N.mu = mu
   physical_parameters_N.flux = flux 
   
   
   physical_parameters_S1 = Physics()
   physical_parameters_S1.N_z = N_z
   physical_parameters_S1.N_phi = N_phi
   physical_parameters_S1.Delta = Delta1
   physical_parameters_S1.t_z = t_z 
   physical_parameters_S1.t_phi = t_phi 
   physical_parameters_S1.mu = mu_1
   physical_parameters_S1.flux = flux 
   
   physical_parameters_S2 = Physics()
   physical_parameters_S2.N_z = N_z
   physical_parameters_S2.N_phi = N_phi
   physical_parameters_S2.Delta = Delta2
   physical_parameters_S2.t_z = t_z 
   physical_parameters_S2.t_phi = t_phi 
   physical_parameters_S2.mu = mu_2
   physical_parameters_S2.flux = flux 
   
   def surface_g(E,physical_parameters):
       # create a dummy Hamiltonian with two layers to get the hopping element beta and the layer element alpha
       
       dummy_params = Physics()
       dummy_params = physical_parameters
       dummy_params.N_z = 2
       
       H_mat = calc_H(dummy_params)

       N_dof_lat = N_phi*2

       alpha = H_mat[:N_dof_lat,:N_dof_lat]
       beta = H_mat[:N_dof_lat,N_dof_lat:2*N_dof_lat]

       err = 1.0
       iter_count = 0
       iter_limit = 100000
       err_limit = 1e-6

       g = np.linalg.inv((E + 1j*eta)*np.eye(alpha.shape[0]) - alpha)
       g_old = np.linalg.inv((E + 1j*eta)*np.eye(alpha.shape[0]) - alpha)
       # iterate over iter_limit iterations or until err < err_limit
       for i in range(iter_limit):
           g = np.linalg.inv((E + 1j*eta)*np.eye(alpha.shape[0]) - alpha - np.dot(np.dot(np.conj(beta.T),g),beta))
           g = 0.5*(g + g_old)

           err = np.linalg.norm(g-g_old)/np.sqrt(np.linalg.norm(g)*np.linalg.norm(g_old))
           g_old = g
           if(err < err_limit):
               #print("Finished at",i,"Error :",err)
               break;
           if(i == (iter_limit - 1)):
               print("iter_limit hit in calculation of surface_g",err)
       return g
   
   g_1 = surface_g(E,physical_parameters_S1)
   g_2 = surface_g(E,physical_parameters_S2)
   
   H_mat = calc_H(physical_parameters_N)
   
   #number of dof in a layer
   N_dof_lat = N_phi*2
   # the hopping element between layers
   beta_layer = H_mat[:N_dof_lat,N_dof_lat:2*N_dof_lat]
   
   # the only non-zero elements in sigma
   sigma_mini_1 = np.dot(np.dot(np.conj(beta_layer.T),g_1),beta_layer)
   sigma_mini_2 = np.dot(np.dot(np.conj(beta_layer.T),g_2),beta_layer)
   
   sigma_1 = np.zeros(H_mat.shape,dtype=np.complex64)
   sigma_1[:N_dof_lat,:N_dof_lat] = sigma_mini_1
   gamma_1 = 1j*(sigma_1 - np.conj(sigma_1).T)
   
   sigma_2 = np.zeros(H_mat.shape,dtype=np.complex64)
   sigma_2[-N_dof_lat:,-N_dof_lat:] = sigma_mini_2
   gamma_2 = 1j*(sigma_2 - np.conj(sigma_2).T)    
   
   def fermi(E,kT):
       return scipy.special.expit(-E/kT)
   
   def generate_fermi_matrix(E,mu,kT):
       return np.array([[fermi(E - mu,kT),0],[0,fermi(E + mu,kT)]])
       
   F1 = np.kron(np.eye(N_phi*N_z),generate_fermi_matrix(E,mu_1-mu,kT))
   F2 = np.kron(np.eye(N_phi*N_z),generate_fermi_matrix(E,mu_2-mu,kT))
   
   sigma_in = np.dot(gamma_1,F1) + np.dot(gamma_2,F2)

   G = np.linalg.inv((E + 1j*eta)*np.eye(H_mat.shape[0]) - H_mat - sigma_1 - sigma_2)
   
   A = 1j*(G - np.conj(G).T)
   
   G_n = np.dot(np.dot(G,sigma_in),np.conj(G).T)
   
   #I_mat = 1j*(np.dot(G_n[:N_dof_lat,N_dof_lat:2*N_dof_lat],beta_layer) \
   #        - np.dot(G_n[N_dof_lat:2*N_dof_lat,:N_dof_lat],beta_layer))
   I_mat = 1j*(np.dot(H_mat,G_n) - np.dot(G_n,H_mat))
   # current = electron current - hole current
   I = np.real(np.trace(I_mat[0:2*N_phi:2,0:2*N_phi:2]-I_mat[1:2*N_phi:2,1:2*N_phi:2]))
   return I

class Physics:
   def __init__(self):
       return


physics = Physics()

phi = np.pi/8

physics.N_z = 2
physics.N_phi = 10
physics.flux = 0.3
physics.t_z = 1
physics.t_phi = 1*np.exp(1j*2*np.pi*physics.flux/physics.N_phi)
physics.Delta1 = 1e-2
physics.Delta2 = 1e-2*np.exp(1j*phi)
physics.mu_1 = 2
physics.mu_2 = 2
physics.eta = 1e-6
physics.kT = 1e-5

I = [current_E(E,physics) for E in np.linspace(-2*physics.Delta1,2*physics.Delta1,100)]
plt.plot(np.linspace(-2*physics.Delta1,2*physics.Delta1,100),I)

import scipy.integrate
def calc_I_phi(x,phi): 
    x.Delta2 = abs(x.Delta2)*np.exp(1j*phi)
    return scipy.integrate.quad(lambda y : current_E(y,x),-15e-3,-5e-3,epsrel=1e-4)
import time
st = time.time()
I_phi = [calc_I_phi(physics,phi) for phi in np.linspace(0,2*np.pi,20)]
plt.plot(np.linspace(0,2*np.pi,len(I_phi)),I_phi)
plt.xlabel('Phase Diff.')
plt.ylabel('Current')
print(time.time()-st)

physics.Delta2

1e-2*np.exp(1j*np.pi/2)

N_f = 10
flux_lin = np.linspace(0.0,5,N_f)
for i in range(N_f):
    physics = Physics()

    physics.N_z = 2
    physics.N_phi = 10
    physics.flux = flux_lin[i]
    physics.t_z = 1
    physics.t_phi = 1*np.exp(1j*2*np.pi*physics.flux/physics.N_phi)
    physics.Delta1 = 1e-2
    physics.Delta2 = 1e-2*np.exp(1j*phi)
    physics.mu_1 = 2
    physics.mu_2 = 2
    physics.eta = 1e-6
    physics.kT = 1e-5

    import scipy.integrate
    def calc_I_phi(x,phi): 
        x.Delta2 = abs(x.Delta2)*np.exp(1j*phi)
        return scipy.integrate.quad(lambda y : current_E(y,x),-15e-3,-5e-3,epsrel=1e-4)
    import time
    st = time.time()
    I_phi = [calc_I_phi(physics,phi) for phi in np.linspace(0,2*np.pi,20)]
    print(physics.flux)
    plt.figure(i+1)
    plt.plot(np.linspace(0,2*np.pi,len(I_phi)),I_phi)
    plt.xlabel('Phase Diff.')
    plt.ylabel('Current')
    print(time.time()-st)



