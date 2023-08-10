# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')

# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

# useful math functions
from math import pi, cos, acos, sqrt

# importing the QISKit
from qiskit import QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

def ch(qProg, a, b):
    """ Controlled-Hadamard gate """
    qProg.h(b)
    qProg.sdg(b)
    qProg.cx(a, b)
    qProg.h(b)
    qProg.t(b)
    qProg.cx(a, b)
    qProg.t(b)
    qProg.h(b)
    qProg.s(b)
    qProg.x(b)
    qProg.s(a)
    return qProg

def cu3(qProg, theta, phi, lambd, c, t):
    """ Controlled-u3 gate """
    qProg.u1((lambd-phi)/2, t)
    qProg.cx(c, t)
    qProg.u3(-theta/2, 0, -(phi+lambd)/2, t)
    qProg.cx(c, t)
    qProg.u3(theta/2, phi, 0, t)
    return qProg

#CHANGE THIS 7BIT 0-1 STRING TO PERFORM EXPERIMENT ON ENCODING 0000000, ..., 1111111
x1234567 = "0101010"

if len(x1234567) != 7 or not("1" in x1234567 or "0" in x1234567):
    raise Exception("x1234567 is a 7-bit 0-1 pattern. Please set it to the correct pattern")
    
#compute the value of rotation angle theta of (3,1)-QRAC 
theta = acos(sqrt(0.5 + sqrt(3.0)/6.0))

#to record the u3 parameters for encoding 000, 010, 100, 110, 001, 011, 101, 111
rotationParams = {"000":(2*theta, pi/4, -pi/4), "010":(2*theta, 3*pi/4, -3*pi/4), 
                  "100":(pi-2*theta, pi/4, -pi/4), "110":(pi-2*theta, 3*pi/4, -3*pi/4), 
                  "001":(2*theta, -pi/4, pi/4), "011":(2*theta, -3*pi/4, 3*pi/4), 
                  "101":(pi-2*theta, -pi/4, pi/4), "111":(pi-2*theta, -3*pi/4, 3*pi/4)}

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config["url"]) # set the APIToken and API url

# Creating registers
# qubits for encoding 7 bits of information with qr[0] kept by the sender
qr = Q_program.create_quantum_register("qr", 3)
# bits for recording the measurement of the qubits qr[1] and qr[2]
cr = Q_program.create_classical_register("cr", 2)

encodingName = "Encode"+x1234567
encodingCircuit = Q_program.create_circuit(encodingName, [qr], [cr])

#Prepare superposition of mixing QRACs of x1...x6 and x7
encodingCircuit.u3(1.187, 0, 0, qr[0])

#Encoding the seventh bit
seventhBit = x1234567[6]
if seventhBit == "1":  #copy qr[0] into qr[1] and qr[2]
    encodingCircuit.cx(qr[0], qr[1])
    encodingCircuit.cx(qr[0], qr[2])
    
#perform controlled-Hadamard qr[0], qr[1], and toffoli qr[0], qr[1] , qr[2]
encodingCircuit = ch(encodingCircuit, qr[0], qr[1])
encodingCircuit.ccx(qr[0], qr[1], qr[2])
#End of encoding the seventh bit

#encode x1...x6 with two (3,1)-QRACS. To do that, we must flip q[0] so that the controlled encoding is executed
encodingCircuit.x(qr[0])

#Encoding the first 3 bits 000, ..., 111 into the second qubit, i.e., (3,1)-QRAC on the second qubit
firstThreeBits = x1234567[0:3]
#encodingCircuit.cu3(*rotationParams[firstThreeBits], qr[0], qr[1])
encodingCircuit = cu3(encodingCircuit, *rotationParams[firstThreeBits], qr[0], qr[1])

#Encoding the second 3 bits 000, ..., 111 into the third qubit, i.e., (3,1)-QRAC on the third qubit
secondThreeBits = x1234567[3:6]
#encodingCircuit.cu3(*rotationParams[secondTreeBits], qr[0], qr[2])
encodingCircuit = cu3(encodingCircuit, *rotationParams[secondThreeBits], qr[0], qr[2])

#end of encoding
encodingCircuit.barrier()

# dictionary for decoding circuits
decodingCircuits = {}
# Quantum circuits for decoding the 1st to 6th bits
for i, pos in enumerate(["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]):
    circuitName = "Decode"+pos
    decodingCircuits[circuitName] = Q_program.create_circuit(circuitName, [qr], [cr])
    if i < 3: #measure 1st, 2nd, 3rd bit
        if pos == "Second": #if pos == "First" we can directly measure
            decodingCircuits[circuitName].h(qr[1])
        elif pos == "Third":
            decodingCircuits[circuitName].u3(pi/2, -pi/2, pi/2, qr[1])
        decodingCircuits[circuitName].measure(qr[1], cr[1])
    else: #measure 4th, 5th, 6th bit
        if pos == "Fifth": #if pos == "Fourth" we can directly measure
            decodingCircuits[circuitName].h(qr[2])
        elif pos == "Sixth":
            decodingCircuits[circuitName].u3(pi/2, -pi/2, pi/2, qr[2])
        decodingCircuits[circuitName].measure(qr[2], cr[1])

        #Quantum circuits for decoding the 7th bit
decodingCircuits["DecodeSeventh"] = Q_program.create_circuit("DecodeSeventh", [qr], [cr])
decodingCircuits["DecodeSeventh"].measure(qr[1], cr[0])
decodingCircuits["DecodeSeventh"].measure(qr[2], cr[1])

#combine encoding and decoding of (7,2)-QRACs to get a list of complete circuits
circuitNames = []
k1 = encodingName
for k2 in decodingCircuits.keys():
    circuitNames.append(k1+k2)
    Q_program.add_circuit(k1+k2, encodingCircuit+decodingCircuits[k2])

print("List of circuit names:", circuitNames) #list of circuit names
#Q_program.get_qasms(circuitNames) #list qasms codes

backend = "local_qasm_simulator"
#backend = "ibmqx2"
shots = 1000
results = Q_program.execute(circuitNames, backend=backend, shots=shots)
for k in ["DecodeFirst", "DecodeSecond", "DecodeThird", "DecodeFourth", "DecodeFifth", "DecodeSixth"]:
    print("Experimental Result of ", encodingName+k)
    plot_histogram(results.get_counts(encodingName+k))

print("Experimental result of ", encodingName+"DecodeSeventh")
plot_histogram(results.get_counts(encodingName+"DecodeSeventh"))

get_ipython().run_line_magic('run', '"../version.ipynb"')



