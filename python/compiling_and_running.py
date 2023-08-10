# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
# Import the QuantumProgram and our configuration
import math
from pprint import pprint

from qiskit import QuantumProgram
import Qconfig

qp = QuantumProgram()
# quantum register for the first circuit
q1 = qp.create_quantum_register('q1', 4)
c1 = qp.create_classical_register('c1', 4)
# quantum register for the second circuit
q2 = qp.create_quantum_register('q2', 2)
c2 = qp.create_classical_register('c2', 2)
# making the first circuits
qc1 = qp.create_circuit('GHZ', [q1], [c1])
qc2 = qp.create_circuit('superpostion', [q2], [c2])
qc1.h(q1[0])
qc1.cx(q1[0], q1[1])
qc1.cx(q1[1], q1[2])
qc1.cx(q1[2], q1[3])
for i in range(4):
    qc1.measure(q1[i], c1[i])
# making the second circuits
qc2.h(q2)
for i in range(2):
    qc2.measure(q2[i], c2[i])
# printing the circuits
print(qp.get_qasm('GHZ'))
print(qp.get_qasm('superpostion'))

qobj = qp.compile(['GHZ','superpostion'], backend='local_qasm_simulator')
qp.get_execution_list(qobj)

qp.get_execution_list(qobj, verbose=True)

qp.get_compiled_configuration(qobj, 'GHZ', )

print(qp.get_compiled_qasm(qobj, 'GHZ'))

# Coupling map 
coupling_map = {0: [1, 2, 3]}
# Place the qubits on a triangle in the bow-tie
initial_layout={("q1", 0): ("q", 0), ("q1", 1): ("q", 1), ("q1", 2): ("q", 2), ("q1", 3): ("q", 3)}

qobj = qp.compile(['GHZ'], backend='local_qasm_simulator', coupling_map=coupling_map, initial_layout=initial_layout)
print(qp.get_compiled_qasm(qobj,'GHZ'))

# Define methods for making QFT circuits
def input_state(circ, q, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(q[j])
        circ.u1(math.pi/float(2**(j)), q[j]).inverse()


def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])

qp = QuantumProgram()
q = qp.create_quantum_register("q", 3)
c = qp.create_classical_register("c", 3)
qft3 = qp.create_circuit("qft3", [q], [c])
input_state(qft3, q, 3)
qft(qft3, q, 3)
for i in range(3):
    qft3.measure(q[i], c[i])
print(qft3.qasm())

result = qp.execute(["qft3"], backend="local_qasm_simulator", shots=1024)
result.get_counts("qft3")

print(result.get_ran_qasm("qft3"))

# Coupling map for ibmqx2 "bowtie"
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}
# Place the qubits on a triangle in the bow-tie
initial_layout={("q", 0): ("q", 2), ("q", 1): ("q", 3), ("q", 2): ("q", 4)}
result2 = qp.execute(["qft3"], backend="local_qasm_simulator", coupling_map=coupling_map, initial_layout=initial_layout, silent=False)
result2.get_counts("qft3")

print(result2.get_ran_qasm("qft3"))

# Place the qubits on a linear segment of the ibmqx3
coupling_map = {0: [1], 1: [2], 2: [3], 3: [14], 4: [3, 5], 6: [7, 11], 7: [10], 8: [7], 9: [8, 10], 11: [10], 12: [5, 11, 13], 13: [4, 14], 15: [0, 14]}
initial_layout={("q", 0): ("q", 0), ("q", 1): ("q", 1), ("q", 2): ("q", 2)}
result3 = qp.execute(["qft3"], backend="local_qasm_simulator", coupling_map=coupling_map, initial_layout=initial_layout)
result3.get_counts("qft3")

print(result3.get_ran_qasm("qft3"))

get_ipython().run_line_magic('run', '"../version.ipynb"')



