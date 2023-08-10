# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
    
# useful additional packages 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram
import Qconfig

# import basic plot tools
from qiskit.tools.visualization import plot_histogram

backend = 'ibmqx2' # the backend to run on
shots = 1024    # the number of shots in the experiment. 
# backend = 'simulator' # the backend test purpose

QPS_SPECS = {
    'circuits': [{
        'name': 'bell',
        'quantum_registers': [{
            'name':'q',
            'size':2
        }],
        'classical_registers': [{
            'name':'c',
            'size':2
        }]}],
}

Q_program = QuantumProgram(specs=QPS_SPECS)
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url'])

# quantum circuit to make Bell state 
bell = Q_program.get_circuit('bell')
q = Q_program.get_quantum_register('q')
c = Q_program.get_classical_register('c')

bell.h(q[0])
bell.cx(q[0],q[1])

# quantum circuit to measure q in standard basis 
measureZZ = Q_program.create_circuit('measureZZ', [q], [c])
measureZZ.measure(q[0], c[0])
measureZZ.measure(q[1], c[1])

# quantum circuit to measure q in superposition basis 
measureXX = Q_program.create_circuit('measureXX', [q], [c])
measureXX.h(q[0])
measureXX.h(q[1])
measureXX.measure(q[0], c[0])
measureXX.measure(q[1], c[1])

# quantum circuit to measure ZX
measureZX = Q_program.create_circuit('measureZX', [q], [c])
measureZX.h(q[0])
measureZX.measure(q[0], c[0])
measureZX.measure(q[1], c[1])

# quantum circuit to measure XZ
measureXZ = Q_program.create_circuit('measureXZ', [q], [c])
measureXZ.h(q[1])
measureXZ.measure(q[0], c[0])
measureXZ.measure(q[1], c[1])

Q_program.add_circuit('bell_measureZX', bell+measureZX )
Q_program.add_circuit('bell_measureXZ', bell+measureXZ )
Q_program.add_circuit('bell_measureZZ', bell+measureZZ )
Q_program.add_circuit('bell_measureXX', bell+measureXX )

circuits = ['bell_measureZZ', 'bell_measureZX', 'bell_measureXX', 'bell_measureXZ']
Q_program.get_qasms(circuits)

result = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

### THIS IS A KNOWN BUG AND WHEN WE FIX THE RETURN FROM THE REAL DEVICE WE WILL ONLY HAVE ONE SET OF OBSERVABLES

observable_first ={'00000': 1, '00001': -1, '00010': 1, '00011': -1}
observable_second ={'00000': 1, '00001': 1, '00010': -1, '00011': -1}
observable_correlated ={'00000': 1, '00001': -1, '00010': -1, '00011': 1}

observable_first_ideal ={'00': 1, '01': -1, '10': 1, '11': -1}
observable_second_ideal ={'00': 1, '01': 1, '10': -1, '11': -1}
observable_correlated_ideal ={'00': 1, '01': -1, '10': -1, '11': 1}

print('IZ = ' + str(result.average_data('bell_measureZZ',observable_first)))
print('ZI = ' + str(result.average_data('bell_measureZZ',observable_second)))
print('ZZ = ' + str(result.average_data('bell_measureZZ',observable_correlated)))

print('IX = ' + str(result.average_data('bell_measureXX',observable_first)))
print('XI = ' + str(result.average_data('bell_measureXX',observable_second)))
print('XX = ' + str(result.average_data('bell_measureXX',observable_correlated)))

print('ZX = ' + str(result.average_data('bell_measureZX',observable_correlated)))
print('XZ = ' + str(result.average_data('bell_measureXZ',observable_correlated)))

CHSH = lambda x : x[0]+x[1]+x[2]-x[3]

backend = 'local_qasm_simulator'
shots = 8192

program = []
xdata=[]

program_end = [measureZZ, measureZX, measureXX, measureXZ]

k = 0
for jj in range(30):
    theta = 2.0*np.pi*jj/30
    bell_middle = QuantumCircuit(q,c)
    bell_middle.ry(theta,q[0])
    for i in range(4):
        program.append('circuit'+str(k))
        Q_program.add_circuit('circuit'+str(k), bell+bell_middle+program_end[i] )
        k += 1
        
    xdata.append(theta)

Q_program.get_qasms(program[0:8])

result = Q_program.execute(program, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

CHSH_data_sim = []
k = 0
for j in range(len(xdata)):
    temp=[]
    for i in range(4): 
        temp.append(result.average_data('circuit'+str(k),observable_correlated_ideal))
        k += 1
    CHSH_data_sim.append(CHSH(temp))

backend = 'ibmqx2'
shots = 1024

program_real = []
xdata_real=[]

k = 0
for jj in range(10):
    theta = 2.0*np.pi*jj/10
    bell_middle = QuantumCircuit(q,c)
    bell_middle.ry(theta,q[0])
    for i in range(4):
        program_real.append('circuit_real'+str(k))
        Q_program.add_circuit('circuit_real'+str(k), bell+bell_middle+program_end[i] )
        k += 1
        
    xdata_real.append(theta)

resultreal = Q_program.execute(program_real, backend=backend, shots=shots, max_credits=3, wait=10, timeout=240, silent=False)

Q_program.get_qasms(program_real[0:8])

CHSH_data_real = []
k = 0
for j in range(len(xdata_real)):
    temp=[]
    for i in range(4): 
        temp.append(resultreal.average_data('circuit_real'+str(k),observable_correlated))
        k += 1
    CHSH_data_real.append(CHSH(temp))

plt.plot(xdata, CHSH_data_sim, 'r-', xdata_real, CHSH_data_real, 'bo')
plt.plot([0, 2*np.pi], [2, 2], 'b-')
plt.plot([0, 2*np.pi], [-2, -2], 'b-')
plt.grid()
plt.ylabel('CHSH', fontsize=20)
plt.xlabel(r'$Y(\theta)$', fontsize=20)
plt.show()

print(CHSH_data_real)

# 2 - qubits 
shots = 8192
backend = 'ibmqx2'

# quantum circuit to make GHZ state
q = Q_program.create_quantum_register('q', 2)
c = Q_program.create_classical_register('c', 2)
ghz = Q_program.create_circuit('ghz', [q], [c])
ghz.h(q[0])
ghz.cx(q[0],q[1])

# quantum circuit to measure q in standard basis 
measureZZ = Q_program.create_circuit('measureZZ', [q], [c])
measureZZ.measure(q[0], c[0])
measureZZ.measure(q[1], c[1])

measureXX = Q_program.create_circuit('measureXX', [q], [c])
measureXX.h(q[0])
measureXX.h(q[1])
measureXX.measure(q[0], c[0])
measureXX.measure(q[1], c[1])

Q_program.add_circuit('ghz_measureZZ', ghz+measureZZ )
Q_program.add_circuit('ghz_measureXX', ghz+measureXX )
circuits = ['ghz_measureZZ', 'ghz_measureXX']
Q_program.get_qasms(circuits)

result2 = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=5, wait=10, timeout=240)
plot_histogram(result2.get_counts('ghz_measureZZ'))
plot_histogram(result2.get_counts('ghz_measureXX'))

# 3 - qubits 
shots = 8192
# quantum circuit to make GHZ state 
q3 = Q_program.create_quantum_register('q3', 3)
c3 = Q_program.create_classical_register('c3', 3)
ghz = Q_program.create_circuit('ghz', [q3], [c3])
ghz.h(q3[0])
ghz.cx(q3[0],q3[1])
ghz.cx(q3[1],q3[2])

# quantum circuit to measure q in standard basis 
measureZZZ = Q_program.create_circuit('measureZZZ', [q3], [c3])
measureZZZ.measure(q3[0], c3[0])
measureZZZ.measure(q3[1], c3[1])
measureZZZ.measure(q3[2], c3[2])

measureXXX = Q_program.create_circuit('measureXXX', [q3], [c3])
measureXXX.h(q3[0])
measureXXX.h(q3[1])
measureXXX.h(q3[2])
measureXXX.measure(q3[0], c3[0])
measureXXX.measure(q3[1], c3[1])
measureXXX.measure(q3[2], c3[2])

Q_program.add_circuit('ghz_measureZZZ', ghz+measureZZZ )
Q_program.add_circuit('ghz_measureXXX', ghz+measureXXX )
circuits = ['ghz_measureZZZ', 'ghz_measureXXX']
Q_program.get_qasms(circuits)
result3 = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=5, wait=10, timeout=240)
plot_histogram(result3.get_counts('ghz_measureZZZ'))
plot_histogram(result3.get_counts('ghz_measureXXX'))

# 4 - qubits 
shots = 8192
# quantum circuit to make GHZ state 
q4 = Q_program.create_quantum_register('q4', 4)
c4 = Q_program.create_classical_register('c4', 4)
ghz = Q_program.create_circuit('ghz', [q4], [c4])
ghz.h(q4[0])
ghz.cx(q4[0],q4[1])
ghz.cx(q4[1],q4[2])
ghz.h(q4[3])
ghz.h(q4[2])
ghz.cx(q4[3],q4[2])
ghz.h(q4[3])
ghz.h(q4[2])

# quantum circuit to measure q in standard basis 
measureZZZZ = Q_program.create_circuit('measureZZZZ', [q4], [c4])
measureZZZZ.measure(q4[0], c4[0])
measureZZZZ.measure(q4[1], c4[1])
measureZZZZ.measure(q4[2], c4[2])
measureZZZZ.measure(q4[3], c4[3])

measureXXXX = Q_program.create_circuit('measureXXXX', [q4], [c4])
measureXXXX.h(q4[0])
measureXXXX.h(q4[1])
measureXXXX.h(q4[2])
measureXXXX.h(q4[3])
measureXXXX.measure(q4[0], c4[0])
measureXXXX.measure(q4[1], c4[1])
measureXXXX.measure(q4[2], c4[2])
measureXXXX.measure(q4[3], c4[3])

Q_program.add_circuit('ghz_measureZZZZ', ghz+measureZZZZ )
Q_program.add_circuit('ghz_measureXXXX', ghz+measureXXXX )
circuits = ['ghz_measureZZZZ', 'ghz_measureXXXX']
Q_program.get_qasms(circuits)
result4 = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=5, wait=10, timeout=480, silent=False)
plot_histogram(result4.get_counts('ghz_measureZZZZ'))
plot_histogram(result4.get_counts('ghz_measureXXXX'))

# quantum circuit to make GHZ state 
q3 = Q_program.create_quantum_register('q3', 3)
c3 = Q_program.create_classical_register('c3', 3)
ghz = Q_program.create_circuit('ghz', [q3], [c3])
ghz.h(q3[0])
ghz.cx(q3[0],q3[1])
ghz.cx(q3[0],q3[2])

# quantum circuit to measure q in standard basis 
measureZZZ = Q_program.create_circuit('measureZZZ', [q3], [c3])
measureZZZ.measure(q3[0], c3[0])
measureZZZ.measure(q3[1], c3[1])
measureZZZ.measure(q3[2], c3[2])

Q_program.add_circuit('ghz_measureZZZ', ghz+measureZZZ )
circuits = ['ghz_measureZZZ']
Q_program.get_qasms(circuits)
result5 = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=5, wait=10, timeout=240, silent=True)
plot_histogram(result5.get_counts('ghz_measureZZZ'))

MerminM = lambda x : x[0]*x[1]*x[2]*x[3]

observable ={'00000': 1, '00001': -1, '00010': -1, '00011': 1, '00100': -1, '00101': 1, '00110': 1, '00111': -1}

# quantum circuit to measure q XXX 
measureXXX = Q_program.create_circuit('measureXXX', [q3], [c3])
measureXXX.h(q3[0])
measureXXX.h(q3[1])
measureXXX.h(q3[2])
measureXXX.measure(q3[0], c3[0])
measureXXX.measure(q3[1], c3[1])
measureXXX.measure(q3[2], c3[2])

# quantum circuit to measure q XYY
measureXYY = Q_program.create_circuit('measureXYY', [q3], [c3])
measureXYY.s(q3[1]).inverse()
measureXYY.s(q3[2]).inverse()
measureXYY.h(q3[0])
measureXYY.h(q3[1])
measureXYY.h(q3[2])
measureXYY.measure(q3[0], c3[0])
measureXYY.measure(q3[1], c3[1])
measureXYY.measure(q3[2], c3[2])

# quantum circuit to measure q YXY
measureYXY = Q_program.create_circuit('measureYXY', [q3], [c3])
measureYXY.s(q3[0]).inverse()
measureYXY.s(q3[2]).inverse()
measureYXY.h(q3[0])
measureYXY.h(q3[1])
measureYXY.h(q3[2])
measureYXY.measure(q3[0], c3[0])
measureYXY.measure(q3[1], c3[1])
measureYXY.measure(q3[2], c3[2])

# quantum circuit to measure q YYX
measureYYX = Q_program.create_circuit('measureYYX', [q3], [c3])
measureYYX.s(q3[0]).inverse()
measureYYX.s(q3[1]).inverse()
measureYYX.h(q3[0])
measureYYX.h(q3[1])
measureYYX.h(q3[2])
measureYYX.measure(q3[0], c3[0])
measureYYX.measure(q3[1], c3[1])
measureYYX.measure(q3[2], c3[2])

Q_program.add_circuit('ghz_measureXXX', ghz+measureXXX )
Q_program.add_circuit('ghz_measureYYX', ghz+measureYYX )
Q_program.add_circuit('ghz_measureYXY', ghz+measureYXY )
Q_program.add_circuit('ghz_measureXYY', ghz+measureXYY )

circuits = ['ghz_measureXXX', 'ghz_measureYYX', 'ghz_measureYXY', 'ghz_measureXYY']
Q_program.get_qasms(circuits)
result6 = Q_program.execute(circuits, backend=backend, shots=shots, max_credits=5, wait=10, timeout=240, silent=False)

temp=[]
temp.append(result6.average_data('ghz_measureXXX',observable))
temp.append(result6.average_data('ghz_measureYYX',observable))
temp.append(result6.average_data('ghz_measureYXY',observable))
temp.append(result6.average_data('ghz_measureXYY',observable))
print(MerminM(temp))

get_ipython().run_line_magic('run', '"../version.ipynb"')



