# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from pprint import pprint

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

zero = np.array([[1],[0]])
one = np.array([[0],[1]])

X = np.array([[0,1],[1,0]])
print(np.dot(X,zero))
print(np.dot(X,one))

backend = 'ibmqx2' # the backend to run on
shots = 1024    # the number of shots in the experiment 

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url


# Creating registers
qr = Q_program.create_quantum_register('qr', 1)
cr = Q_program.create_classical_register('cr', 1)

# Quantum circuit ground 
qc_ground = Q_program.create_circuit('ground', [qr], [cr])
qc_ground.measure(qr[0], cr[0])

# Quantum circuit excited 
qc_excited = Q_program.create_circuit('excited', [qr], [cr])
qc_excited.x(qr)
qc_excited.measure(qr[0], cr[0])


circuits = ['ground', 'excited']

Q_program.get_qasms(circuits)

result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

plot_histogram(result.get_counts('ground'))

plot_histogram(result.get_counts('excited'))

# Quantum circuit superposition 
qc_superposition = Q_program.create_circuit('superposition', [qr], [cr])
qc_superposition.h(qr)
qc_superposition.measure(qr[0], cr[0])

circuits = ['superposition']
result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

plot_histogram(result.get_counts('superposition'))

# Quantum circuit two Hadamards 
qc_twohadamard = Q_program.create_circuit('twohadamard', [qr], [cr])
qc_twohadamard.h(qr)
qc_twohadamard.barrier()
qc_twohadamard.h(qr)
qc_twohadamard.measure(qr[0], cr[0])

circuits = ['twohadamard']
result= Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

plot_histogram(result.get_counts('twohadamard'))

state_overlap = lambda state1, state2: np.absolute(np.dot(state1.conj().T,state2))**2

Hadamard = np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)
psi1 = np.dot(Hadamard,zero)
P0 = state_overlap(zero,psi1)
P1 = state_overlap(one,psi1)
plot_histogram({'0' : P0, '1' : P1})

print(np.dot(Hadamard,zero))
print(np.dot(Hadamard,one))

backend = 'ibmqx2' # the backend to run on
shots = 1024    # the number of shots in the experiment 

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# Creating registers
q2 = Q_program.create_quantum_register("q2", 2)
c2 = Q_program.create_classical_register("c2", 2)

# quantum circuit to make an entangled bell state 
bell = Q_program.create_circuit("bell", [q2], [c2])
bell.h(q2[0])
bell.cx(q2[0], q2[1])

# quantum circuit to measure q0 in the standard basis
measureIZ = Q_program.create_circuit("measureIZ", [q2], [c2])
measureIZ.measure(q2[0], c2[0])

# quantum circuit to measure q0 in the superposition basis 
measureIX = Q_program.create_circuit("measureIX", [q2], [c2])
measureIX.h(q2[0])
measureIX.measure(q2[0], c2[0])

# quantum circuit to measure q1 in the standard basis
measureZI = Q_program.create_circuit("measureZI", [q2], [c2])
measureZI.measure(q2[1], c2[1])

# quantum circuit to measure q1 in the superposition basis 
measureXI = Q_program.create_circuit("measureXI", [q2], [c2])
measureXI.h(q2[1])
measureXI.measure(q2[1], c2[1])

# quantum circuit to measure q in the standard basis 
measureZZ = Q_program.create_circuit("measureZZ", [q2], [c2])
measureZZ.measure(q2[0], c2[0])
measureZZ.measure(q2[1], c2[1])

# quantum circuit to measure q in the superposition basis 
measureXX = Q_program.create_circuit("measureXX", [q2], [c2])
measureXX.h(q2[0])
measureXX.h(q2[1])
measureXX.measure(q2[0], c2[0])
measureXX.measure(q2[1], c2[1])

Q_program.add_circuit("bell_measureIZ", bell+measureIZ )
Q_program.add_circuit("bell_measureIX", bell+measureIX )
Q_program.add_circuit("bell_measureZI", bell+measureZI )
Q_program.add_circuit("bell_measureXI", bell+measureXI )
Q_program.add_circuit("bell_measureZZ", bell+measureZZ )
Q_program.add_circuit("bell_measureXX", bell+measureXX )

circuits = ["bell_measureIZ", "bell_measureIX", "bell_measureZI", "bell_measureXI", "bell_measureZZ", "bell_measureXX"]
Q_program.get_qasms(circuits)

result = Q_program.execute(circuits[0:2], backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)
plot_histogram(result.get_counts("bell_measureIZ"))

result.get_data("bell_measureIZ")

plot_histogram(result.get_counts("bell_measureIX"))

result = Q_program.execute(circuits[2:4], backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)
plot_histogram(result.get_counts("bell_measureZI"))
plot_histogram(result.get_counts("bell_measureXI"))

result = Q_program.execute(circuits[4:6], backend=backend, shots=shots, max_credits=3, wait=10, timeout=240,silent=False)

plot_histogram(result.get_counts("bell_measureZZ"))

plot_histogram(result.get_counts("bell_measureXX"))

# Creating registers
q2 = Q_program.create_quantum_register("q2", 2)
c2 = Q_program.create_classical_register("c2", 2)

# quantum circuit to make a mixed state 
mixed1 = Q_program.create_circuit("mixed1", [q2], [c2])
mixed2 = Q_program.create_circuit("mixed2", [q2], [c2])
mixed2.x(q2)
mixed1.measure(q2[0], c2[0])
mixed1.measure(q2[1], c2[1])
mixed2.measure(q2[0], c2[0])
mixed2.measure(q2[1], c2[1])
mixed_state = ["mixed1", "mixed2"]
result = Q_program.execute(mixed_state, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)
counts1 = result.get_counts(mixed_state[0])
counts2 = result.get_counts(mixed_state[1])
from collections import Counter
ground = Counter(counts1)
excited = Counter(counts2)
plot_histogram(ground+excited)

get_ipython().run_line_magic('run', '"../version.ipynb"')



