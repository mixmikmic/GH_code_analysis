# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from collections import Counter #Use this to convert results from list to dict for histogram


# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

device = 'ibmqx2' # the device to run on
#device = 'local_qasm_simulator' # uncomment to run on the simulator
N = 50 # Number of bombs
steps = 20 # Number of steps for the algorithm, limited by maximum circuit depth
eps = np.pi / steps # Algorithm parameter, small

QPS_SPECS = {
    "name": "IFM",
    "circuits": [{
        "name": "IFM_gen", # Prototype circuit for bomb generation
        "quantum_registers": [{
            "name":"q_gen",
            "size":1
        }],
        "classical_registers": [{
            "name":"c_gen",
            "size":1
        }]},
        {"name": "IFM_meas", # Prototype circuit for bomb measurement
        "quantum_registers": [{
            "name":"q",
            "size":2
        }],
        "classical_registers": [{
            "name":"c",
            "size":steps+1
        }]}]
}

Q_program = QuantumProgram(specs=QPS_SPECS)
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"])

# Quantum circuits to generate bombs
circuits = ["IFM_gen"+str(i) for i in range(N)]
# NB: Can't have more than one measurement per circuit
for circuit in circuits:
    q_gen = Q_program.get_quantum_register("q_gen")
    c_gen = Q_program.get_classical_register('c_gen')
    IFM = Q_program.create_circuit(circuit, [q_gen], [c_gen])
    IFM.h(q_gen[0]) #Turn the qubit into |0> + |1>
    IFM.measure(q_gen[0], c_gen[0])
_ = Q_program.get_qasms(circuits) # Suppress the output

result = Q_program.execute(circuits, device, shots=1, max_credits=5, wait=10, timeout=240) # Note that we only want one shot
bombs = []
for circuit in circuits:
    for key in result.get_counts(circuit): # Hack, there should only be one key, since there was only one shot
        bombs.append(int(key))
#print(', '.join(('Live' if bomb else 'Dud' for bomb in bombs))) # Uncomment to print out "truth" of bombs
plot_histogram(Counter(('Live' if bomb else 'Dud' for bomb in bombs))) #Plotting bomb generation results

device = 'local_qasm_simulator' #Running on the simulator
circuits = ["IFM_meas"+str(i) for i in range(N)]
#Creating one measurement circuit for each bomb
for i in range(N):
    bomb = bombs[i]
    q = Q_program.get_quantum_register("q")
    c = Q_program.get_classical_register('c')
    IFM = Q_program.create_circuit(circuits[i], [q], [c])
    for step in range(steps):
        IFM.ry(eps, q[0]) #First we rotate the control qubit by epsilon
        if bomb: #If the bomb is live, the gate is a controlled X gate
            IFM.cx(q[0],q[1])
        #If the bomb is a dud, the gate is a controlled identity gate, which does nothing
        IFM.measure(q[1], c[step]) #Now we measure to collapse the combined state
    IFM.measure(q[0], c[steps])
Q_program.get_qasms(circuits)
result = Q_program.execute(circuits, device, shots=1, max_credits=5, wait=10, timeout=240)

def get_status(counts):
    # Return whether a bomb was a dud, was live but detonated, or was live and undetonated
    # Note that registers are returned in reversed order
    for key in counts:
        if '1' in key[1:]:
             #If we ever measure a '1' from the measurement qubit (q1), the bomb was measured and will detonate
            return '!!BOOM!!'
        elif key[0] == '1':
            #If the control qubit (q0) was rotated to '1', the state never entangled because the bomb was a dud
            return 'Dud'
        else:
            #If we only measured '0' for both the control and measurement qubit, the bomb was live but never set off
            return 'Live'


results = {'Live': 0, 'Dud': 0, "!!BOOM!!": 0}
for circuit in circuits:
    status = get_status(result.get_counts(circuit))
    results[status] += 1
plot_histogram(results)

get_ipython().run_line_magic('run', '"../version.ipynb"')



