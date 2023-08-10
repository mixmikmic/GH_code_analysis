# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from math import pi

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

backend = 'local_qasm_simulator' # the device to run on
shots = 1024    # the number of shots in the experiment 

#to record the rotation number for encoding 00, 10, 11, 01
rotationNumbers = {"00":1, "10":3, "11":5, "01":7}

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# Creating registers
# qubit for encoding 2 bits of information
qr = Q_program.create_quantum_register("qr", 1)
# bit for recording the measurement of the qubit
cr = Q_program.create_classical_register("cr", 1)

# dictionary for encoding circuits
encodingCircuits = {}
# Quantum circuits for encoding 00, 10, 11, 01
for bits in ("00", "01", "10", "11"):
    circuitName = "Encode"+bits
    encodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    encodingCircuits[circuitName].u3(rotationNumbers[bits]*pi/4.0, 0, 0, qr[0])
    encodingCircuits[circuitName].barrier()

# dictionary for decoding circuits
decodingCircuits = {}
# Quantum circuits for decoding the first and second bit
for pos in ("First", "Second"):
    circuitName = "Decode"+pos
    decodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    if pos == "Second": #if pos == "First" we can directly measure
        decodingCircuits[circuitName].h(qr[0])
    decodingCircuits[circuitName].measure(qr[0], cr[0])

#combine encoding and decoding of QRACs to get a list of complete circuits
circuitNames = []
for k1 in encodingCircuits.keys():
    for k2 in decodingCircuits.keys():
        circuitNames.append(k1+k2)
        Q_program.add_circuit(k1+k2, encodingCircuits[k1]+decodingCircuits[k2])

print("List of circuit names:", circuitNames) #list of circuit names
#Q_program.get_qasms(circuitNames) #list qasms codes

results = Q_program.execute(circuitNames, backend=backend, shots=shots)
print("Experimental Result of Encode01DecodeFirst")
plot_histogram(results.get_counts("Encode01DecodeFirst"))  #We should measure "0" with probability 0.85
print("Experimetnal Result of Encode01DecodeSecond")
plot_histogram(results.get_counts("Encode01DecodeSecond")) #We should measure "1" with probability 0.85
print("Experimental Result of Encode11DecodeFirst")
plot_histogram(results.get_counts("Encode11DecodeFirst"))  #We should measure "1" with probability 0.85
print("Experimental Result of Encode11DecodeSecond")
plot_histogram(results.get_counts("Encode11DecodeSecond")) #We should measure "1" with probability 0.85

backend = "ibmqx2" 
#backend = "local_qasm_simulator"
results = Q_program.execute(["Encode01DecodeFirst", "Encode01DecodeSecond"], backend=backend, shots=shots)
print("Experimental Result of Encode01DecodeFirst")
plot_histogram(results.get_counts("Encode01DecodeFirst"))  #We should measure "0" with probability 0.85
print("Experimental Result of Encode01DecodeSecond")
plot_histogram(results.get_counts("Encode01DecodeSecond")) #We should measure "1" with probability 0.85

backend = 'local_qasm_simulator' # the device to run on
shots = 1024    # the number of shots in the experiment 
from math import sqrt, cos, acos

#compute the value of theta
theta = acos(sqrt(0.5 + sqrt(3.0)/6.0))

#to record the u3 parameters for encoding 000, 010, 100, 110, 001, 011, 101, 111
rotationParams = {"000":(2*theta, pi/4, -pi/4), "010":(2*theta, 3*pi/4, -3*pi/4), 
                  "100":(pi-2*theta, pi/4, -pi/4), "110":(pi-2*theta, 3*pi/4, -3*pi/4), 
                  "001":(2*theta, -pi/4, pi/4), "011":(2*theta, -3*pi/4, 3*pi/4), 
                  "101":(pi-2*theta, -pi/4, pi/4), "111":(pi-2*theta, -3*pi/4, 3*pi/4)}

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# Creating registers
# qubit for encoding 3 bits of information
qr = Q_program.create_quantum_register("qr", 1)
# bit for recording the measurement of the qubit
cr = Q_program.create_classical_register("cr", 1)

# dictionary for encoding circuits
encodingCircuits = {}
# Quantum circuits for encoding 000, ..., 111
for bits in rotationParams.keys():
    circuitName = "Encode"+bits
    encodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    encodingCircuits[circuitName].u3(*rotationParams[bits], qr[0])
    encodingCircuits[circuitName].barrier()

# dictionary for decoding circuits
decodingCircuits = {}
# Quantum circuits for decoding the first, second and third bit
for pos in ("First", "Second", "Third"):
    circuitName = "Decode"+pos
    decodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    if pos == "Second": #if pos == "First" we can directly measure
        decodingCircuits[circuitName].h(qr[0])
    elif pos == "Third":
        decodingCircuits[circuitName].u3(pi/2, -pi/2, pi/2, qr[0])
    decodingCircuits[circuitName].measure(qr[0], cr[0])

#combine encoding and decoding of QRACs to get a list of complete circuits
circuitNames = []
for k1 in encodingCircuits.keys():
    for k2 in decodingCircuits.keys():
        circuitNames.append(k1+k2)
        Q_program.add_circuit(k1+k2, encodingCircuits[k1]+decodingCircuits[k2])

print("List of circuit names:", circuitNames) #list of circuit names
#Q_program.get_qasms(circuitNames) #list qasms codes

results = Q_program.execute(circuitNames, backend=backend, shots=shots)
print("Experimental Result of Encode010DecodeFirst")
plot_histogram(results.get_counts("Encode010DecodeFirst"))  #We should measure "0" with probability 0.78
print("Experimental Result of Encode010DecodeSecond")
plot_histogram(results.get_counts("Encode010DecodeSecond")) #We should measure "1" with probability 0.78
print("Experimental Result of Encode010DecodeThird")
plot_histogram(results.get_counts("Encode010DecodeThird"))  #We should measure "0" with probability 0.78

backend = "ibmqx2"
#backend = "local_qasm_simulator"
results = Q_program.execute(["Encode010DecodeFirst", "Encode010DecodeSecond", "Encode010DecodeThird"], backend=backend, shots=shots)
print("Experimental Result of Encode010DecodeFirst")
plot_histogram(results.get_counts("Encode010DecodeFirst"))  #We should measure "0" with probability 0.78
print("Experimental Result of Encode010DecodeSecond")
plot_histogram(results.get_counts("Encode010DecodeSecond")) #We should measure "1" with probability 0.78
print("Experimental Result of Encode010DecodeThird")
plot_histogram(results.get_counts("Encode010DecodeThird"))  #We should measure "0" with probability 0.78

get_ipython().run_line_magic('run', '"../version.ipynb"')



