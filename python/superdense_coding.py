# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
# Importing QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig

# Import basic plotting tools
from qiskit.tools.visualization import plot_histogram

# Quantum program setup 
Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# Creating registers
q = Q_program.create_quantum_register("q", 2)
c = Q_program.create_classical_register("c", 2)

# Quantum circuit to make the shared entangled state 
superdense = Q_program.create_circuit("superdense", [q], [c])
superdense.h(q[0])
superdense.cx(q[0], q[1])

# For 00, do nothing

# For 01, apply $X$
#shared.x(q[0])

# For 01, apply $Z$
#shared.z(q[0])

# For 11, apply $XZ$
superdense.z(q[0]) 
superdense.x(q[0])
superdense.barrier()

superdense.cx(q[0], q[1])
superdense.h(q[0])
superdense.measure(q[0], c[0])
superdense.measure(q[1], c[1])

circuits = ["superdense"]
print(Q_program.get_qasms(circuits)[0])

backend = 'ibmqx2'  # the device to run on
shots = 1024       # the number of shots in the experiment 

result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240)

plot_histogram(result.get_counts("superdense"))

get_ipython().run_line_magic('run', '"../version.ipynb"')



