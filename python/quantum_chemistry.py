# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from scipy import linalg as la
from functools import partial

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram


# import optimization tools
from qiskit.tools.apps.optimization import trial_circuit_ryrz, SPSA_optimization, SPSA_calibration
from qiskit.tools.apps.optimization import Hamiltonian_from_file, make_Hamiltonian
from qiskit.tools.apps.optimization import eval_hamiltonian, group_paulis

# Ignore warnings due to chopping of small imaginary part of the energy 
import warnings
warnings.filterwarnings('ignore')
                

n=2
m=6
device='local_qasm_simulator'

initial_theta=np.random.randn(2*n*m)
entangler_map={0: [1]} # the map of two-qubit gates with control at key and target at values
shots=1
max_trials=100
ham_name='H2/H2Equilibrium.txt'

# Exact Energy
pauli_list=Hamiltonian_from_file(ham_name)
H=make_Hamiltonian(pauli_list)
exact=np.amin(la.eig(H)[0]).real
print('The exact ground state energy is:')
print(exact)
pauli_list_grouped=group_paulis(pauli_list)

# Optimization
Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken,Qconfig.config["url"])



def cost_function(Q_program,H,n,m,entangler_map,shots,device,theta):
    
    return eval_hamiltonian(Q_program,H,trial_circuit_ryrz(n,m,theta,entangler_map,None,False),shots,device).real


initial_c=0.01
target_update=2*np.pi*0.1
save_step = 20

if shots ==1:
    SPSA_params=SPSA_calibration(partial(cost_function,Q_program,H,n,m,entangler_map,
                                         shots,device),initial_theta,initial_c,target_update,25)
    output=SPSA_optimization(partial(cost_function,Q_program,H,n,m,entangler_map,shots,device),
                         initial_theta,SPSA_params,max_trials,save_step,1);
else:
    SPSA_params=SPSA_calibration(partial(cost_function,Q_program,pauli_list_grouped,n,m,entangler_map,
                                         shots,device),initial_theta,initial_c,target_update,25)
    output=SPSA_optimization(partial(cost_function,Q_program,pauli_list_grouped,n,m,entangler_map,shots,device),
                         initial_theta,SPSA_params,max_trials,save_step,1);


plt.plot(np.arange(0, max_trials,save_step),output[2],label='E(theta_plus)')
plt.plot(np.arange(0, max_trials,save_step),output[3],label='E(theta_minus)')
plt.plot(np.arange(0, max_trials,save_step),np.ones(max_trials//save_step)*output[0],label='Final Energy')
plt.plot(np.arange(0, max_trials,save_step),np.ones(max_trials//save_step)*exact,label='Exact Energy')
plt.legend()
plt.xlabel('Trial state')
plt.ylabel('Energy')

# MOLECULE PARAMETERS

molecule='H2' # name of the molecule: options are H2 or LiH

if molecule=='H2':
    n=2 # qubits
    Z1=1
    Z2=1
    min_distance=.2
    max_distance=4
    number_of_points=39
elif molecule=='LiH':
    n=4 # qubits
    Z1=1
    Z2=3
    min_distance=.5
    max_distance=5
    number_of_points=46
    
# OPTIMIZATION PARAMETERS 

run_optimization=True # Choose True or False. False just plots the exact potential energy surface

def cost_function(Q_program,H,n,m,entangler_map,shots,device,theta):
    
    return eval_hamiltonian(Q_program,H,trial_circuit_ryrz(n,m,theta,entangler_map,None,False),shots,device).real


m=3 # depth (number of layers of sq gates - # entanglers is this number -1)
device='local_qasm_simulator'
initial_theta=np.random.randn(2*n*m)  # initial angles 
entangler_map={0: [1]} # the map of two-qubit gates with control at key and target at values
shots=1
max_trials=200

initial_c = 0.01
target_update = 2*np.pi*0.1
save_step = 250 #Setting larger than max_trials to suppress the output at every function call

####################     COMPUTING THE POTENTIAL ENERGY SURFACE      ###################

mol_distance=np.zeros(number_of_points)
coulomb_repulsion=np.zeros(number_of_points)
electr_energy=np.zeros(number_of_points)
electr_energy_optimized=np.zeros(number_of_points)

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken,Qconfig.config["url"])

for i in range(number_of_points):

    # %%%%%%%%%%%%% Coulomb Repulsion For Diatomic Molecules %%%%%%%%%%%%%%%%%%%%%%
        mol_distance[i]=np.around((min_distance+(max_distance-min_distance)*i/(number_of_points-1)),2)
        
        distance=mol_distance[i]/0.529177
        coulomb_repulsion[i]=Z1*Z2/distance
        
    # exact diagonalization
        ham_text=molecule+'/PESMap'+str(i)+'atdistance'+str(mol_distance[i])+'.txt'
        pauli_list=Hamiltonian_from_file(ham_text)
        H=make_Hamiltonian(pauli_list)
        pauli_list_grouped = group_paulis(pauli_list)
        eigen=la.eig(H)
        electr_energy[i]=np.amin(eigen[0])
           
    # optimization
    
        if run_optimization:
            print('\nOPTIMIZING HAMILTONIAN # '+str(i)+' AT INTERATOMIC DISTANCE ' + str(mol_distance[i]) + ' ANGSTROM\n')
            
            
            if shots !=1:
                H=group_paulis(pauli_list)
         
        
            SPSA_params = SPSA_calibration(partial(cost_function,Q_program,H,n,m,entangler_map,
                                           shots,device),initial_theta,initial_c,target_update,25)
            
            electr_energy_optimized[i] = SPSA_optimization(partial(cost_function,Q_program,H,n,m,entangler_map,shots,device),
                                                           initial_theta,SPSA_params,max_trials,save_step,1)[0];

    
    

plt.plot(mol_distance,electr_energy+coulomb_repulsion,label='Exact')
if run_optimization:
    plt.plot(mol_distance,electr_energy_optimized+coulomb_repulsion,label='Optimized') 
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()

get_ipython().run_line_magic('run', '"../version.ipynb"')



