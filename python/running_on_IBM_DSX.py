# This cell does some preparatory work to run this Jupyter notebook using 
# IBM Data Science Experience (datascience.ibm.com). All the other cells remain the same as in the original example. 

# --------------------------------------------------------------------------------------------------
# IMPORTANT NOTES: 
# 1) Log on to your QX account and get your API token. You can get it from the Accounts page 
# (the top-right icon on QX website) Copy-Paste the API token below.BE SURE TO ENCLOSE IN QUOTES ("")
# 2) If you are sharing this notebook, MAKE SURE TO REMOVE the API token string before sharing! 
#   Alternatively, you can modify the code below and separately create and save Qconfig.py file with 
#   API token.
# --------------------------------------------------------------------------------------------------
QX_API_TOKEN = "PUT_YOUR_API_TOKEN_HERE"

import sys
import os

try:
    assert(sys.version_info.major > 2)
except AssertionError:          
    print("This code requires Python 3.5 or beyond. Your version: {}.{}".format(sys.version_info.major,sys.version_info.minor))
    raise

# DO NOT CHANGE THE FOLLOWING assertion
try:
    assert(QX_API_TOKEN != "PUT_YOUR_API_TOKEN_HERE")
except AssertionError:
    print("Update the value of QX_API_TOKEN first!")
    raise
    
# importlib.reload is used to reload the Qconfig module. Fixes any update problems with Qconfig.APItoken 
import importlib

# We need visibility to Qconfig.py module add current dir your sys.path
sys.path.append("./")

# Install qiskit package
get_ipython().system('pip install qiskit')
get_ipython().system('rm -rf Qconfig.py ')
get_ipython().system('touch Qconfig.py')

apitoken_str = 'APItoken=\"{}\"'.format(QX_API_TOKEN)
config_str = "config = { \"url\": 'https://quantumexperience.ng.bluemix.net/api'}"

with open('Qconfig.py', 'w') as fil:
    fil.write(apitoken_str + "\n")
    fil.write(config_str + "\n")
    
# Now import all the modules from qiskit such as QuantumProgram, QuantumCircuit, QuantumRegister etc.
# Optionally, you can modify the qiskit import line below to do specific module imports. 
from qiskit import *
import Qconfig

try:
    # Reload Qconfig module again to make sure the APItoken is updated
    importlib.reload(Qconfig)
except NameError as e:
    print("Name error: ", e.args)
except:
    print("unable to reload the module, continuing the execution..")
    
print("Qconfig.APItoken = ", Qconfig.APItoken)

# Creating Programs
# create your first QuantumProgram object instance.
qp = QuantumProgram()

# Creating Registers
# create your first Quantum Register called "qr" with 2 qubits 
qr = qp.create_quantum_register('qr', 2)
# create your first Classical Register  called "cr" with 2 bits
cr = qp.create_classical_register('cr', 2)

# Creating Circuits
# create your first Quantum Circuit called "qc" involving your Quantum Register "qr"
# and your Classical Register "cr"
qc = qp.create_circuit('qc', [qr], [cr])

Q_SPECS = {
    'circuits': [{
        'name': 'Circuit',
        'quantum_registers': [{
            'name': 'qr',
            'size': 4
        }],
        'classical_registers': [{
            'name': 'cr',
            'size': 4
        }]}],
}

qp = QuantumProgram(specs=Q_SPECS)

# Get the components.

# get the circuit by Name
circuit = qp.get_circuit('Circuit')

# get the Quantum Register by Name
quantum_r = qp.get_quantum_register('qr')

# get the Classical Register by Name
classical_r = qp.get_classical_register('cr')

# Pauli X gate to qubit 1 in the Quantum Register "qr" 
circuit.x(quantum_r[1])

# Pauli Y gate to qubit 2 in the Quantum Register "qr" 
circuit.y(quantum_r[2])

# Pauli Z gate to qubit 3 in the Quantum Register "qr" 
circuit.z(quantum_r[3])

# CNOT (Controlled-NOT) gate from qubit 0 to qubit 2
circuit.cx(quantum_r[0], quantum_r[2])

# add a barrier to your circuit
circuit.barrier()

# H (Hadamard) gate to qubit 0 in the Quantum Register "qr" 
circuit.h(quantum_r[0])

# S Phase gate to qubit 0
circuit.s(quantum_r[0])

# T Phase gate to qubit 1
circuit.t(quantum_r[1])

# identity gate to qubit 1
circuit.iden(quantum_r[1])

# first physical gate: u1(lambda) to qubit 0
circuit.u1(0.3, quantum_r[0])

# second physical gate: u2(phi,lambda) to qubit 1
circuit.u2(0.3, 0.2, quantum_r[1])

# second physical gate: u3(theta,phi,lambda) to qubit 2
circuit.u3(0.3, 0.2, 0.1, quantum_r[2])

# rotation around the x-axis to qubit 0
circuit.rx(0.2, quantum_r[0])

# rotation around the y-axis to qubit 1
circuit.ry(0.2, quantum_r[1])

# rotation around the z-axis to qubit 2
circuit.rz(0.2, quantum_r[2])

# Classical if, from qubit2 gate Z to classical bit 1
# circuit.z(quantum_r[2]).c_if(classical_r, 0)

# measure gate from qubit 0 to classical bit 0
circuit.measure(quantum_r[0], classical_r[0])
circuit.measure(quantum_r[1], classical_r[1])
circuit.measure(quantum_r[2], classical_r[2])

qp.get_circuit_names()

# QASM from a program

QASM_source = qp.get_qasm('Circuit')

print(QASM_source)

backend = 'local_qasm_simulator' 
circuits = ['Circuit']  # Group of circuits to execute

qobj=qp.compile(circuits, backend) # Compile your program

result = qp.run(qobj, wait=2, timeout=240)
print(result)

result.get_counts('Circuit')

ran_qasm = result.get_ran_qasm('Circuit')

print(ran_qasm)

out = qp.execute(circuits, backend, wait=2, timeout=240)
print(out)

backend = 'ibmqx2'   # Backend where you execute your program; in this case, on the Real Quantum Chip online 
circuits = ['Circuit']   # Group of circuits to execute
shots = 1024           # Number of shots to run the program (experiment); maximum is 8192 shots.
max_credits = 3          # Maximum number of credits to spend on executions. 
qp.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url

result_real = qp.execute(circuits, backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

result_real2 = qp.execute(circuits, backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=True)

get_ipython().run_line_magic('run', '"../version.ipynb"')

result_real.get_counts('Circuit')

result_real2.get_counts('Circuit')



