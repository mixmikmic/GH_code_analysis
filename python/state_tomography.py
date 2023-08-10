# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
import numpy as np
    
# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig

# import tomography libary
import qiskit.tools.qcvv.tomography as tomo

# useful additional packages 
from qiskit.tools.visualization import plot_state, plot_histogram
from qiskit.tools.qi.qi import state_fidelity, concurrence, purity, outer

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url

# Creating registers
qr = Q_program.create_quantum_register('qr', 2)
cr = Q_program.create_classical_register('cr', 2)

# quantum circuit to make an entangled bell state 
bell = Q_program.create_circuit('bell', [qr], [cr])
bell.h(qr[0])
bell.cx(qr[0], qr[1])

bell_result = Q_program.execute(['bell'], backend='local_qasm_simulator', shots=1)
bell_psi = bell_result.get_data('bell')['quantum_state']
bell_rho = outer(bell_psi) # construct the density matrix from the state vector

# plot the state
plot_state(bell_rho,'paulivec')

rho_mixed = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])/2
plot_state(rho_mixed, 'paulivec')

# Qubits being measured
bell_qubits = [0,1]
# Construct the state tomography measurement circuits in QP
bell_tomo_circuits = tomo.build_state_tomography_circuits(Q_program, 'bell', bell_qubits, qr, cr)
for c in bell_tomo_circuits:
    print(c)

# Use the local simulator
backend = 'local_qasm_simulator'

# Take 1000 shots for each measurement basis
shots = 5000

# Run the simulation
bell_tomo_result = Q_program.execute(bell_tomo_circuits, backend=backend, shots=shots)
print(bell_tomo_result)

bell_tomo_data = tomo.state_tomography_data(bell_tomo_result, 'bell', bell_qubits)

rho_fit = tomo.fit_tomography_data(bell_tomo_data)

# target state is (|00>+|11>)/sqrt(2)
target = np.array([1., 0., 0., 1.]/np.sqrt(2.))

# calculate fidelity, concurrence and purtity of fitted state
F_fit = state_fidelity(rho_fit, [0.707107, 0, 0, 0.707107])
con = concurrence(rho_fit)
pur = purity(rho_fit)

# plot 
plot_state(rho_fit, 'paulivec')
print('Fidelity =', F_fit)
print('concurrence = ', str(con))
print('purity = ', str(pur))

# Use the IBM Quantum Experience
backend = 'ibmqx2'
# Take 1000 shots for each measurement basis
# Note: reduce this number for larger number of qubits
shots = 1000
# set max credits
max_credits = 5

# Run the simulation
bellqx_tomo_results = Q_program.execute(bell_tomo_circuits, backend=backend, shots=shots,
                           max_credits=max_credits, silent=False,wait=20, timeout=240)
print(bellqx_tomo_results)

bellqx_tomo_data = tomo.state_tomography_data(bellqx_tomo_results, 'bell', bell_qubits)

rho_fit_real = tomo.fit_tomography_data(bellqx_tomo_data)

F_fit_real = state_fidelity(rho_fit_real, [0.707107, 0, 0, 0.707107])
plot_state(rho_fit_real, 'paulivec')
print('Fidelity with ideal state')
print('F =', F_fit_real)

# calculate concurrence and purity
con = concurrence(rho_fit_real)
pur = purity(rho_fit_real)
print('concurrence = ', str(con))
print('purity = ', str(pur))

get_ipython().run_line_magic('run', '"../version.ipynb"')



