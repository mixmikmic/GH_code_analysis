# Checking the version of PYTHON; we only support > 3.5
import sys
if sys.version_info < (3,5):
    raise Exception('Please use Python version 3.5 or greater.')
import sys
import sys
from qiskit import QuantumProgram
import Qconfig
import numpy as np
from scipy.optimize import curve_fit
from qiskit.tools.qcvv.fitters import exp_fit_fun, osc_fit_fun, plot_coherence

# function for padding with QId gates
def pad_QId(circuit,N,qr):
    # circuit to add to, N= number of QId gates to add, qr=qubit reg
    for ii in range(N):
        circuit.iden(qr)
    return circuit    

# backend and token settings
backend = 'ibmqx4' # the device to run on
# device = 'simulator'
shots = 1024    # the number of shots in the experiment 

Q_program = QuantumProgram()
Q_program.set_api(Qconfig.APItoken, Qconfig.config['url']) # set the APIToken and API url

# Select qubit to measure T1 on
qubit=3

# Creating registers
qr = Q_program.create_quantum_register('qr', 5)
cr = Q_program.create_classical_register('cr', 5)

# the delay times are all set in terms of single-qubit gates
# so we need to caluclate the time from these parameters

params = Q_program.get_backend_parameters(backend)['qubits'][qubit]
pulse_length=params['gateTime']['value'] # single-qubit gate time 
buffer_length=params['buffer']['value'] # spacing between pulses
unit = params['gateTime']['unit']

steps=10
gates_per_step=80
max_gates=(steps-1)*gates_per_step+1
tot_length=buffer_length+pulse_length
time_per_step=gates_per_step*tot_length
qc_dict={}
for ii in range(steps):
    step_num='step_%s'%(str(ii))
    qc_dict.update({step_num:Q_program.create_circuit(step_num, [qr], [cr])})
    qc_dict[step_num].x(qr[qubit])
    qc_dict[step_num]=pad_QId(qc_dict[step_num],gates_per_step*ii,qr[qubit])
    qc_dict[step_num].measure(qr[qubit], cr[qubit])

circuits=list(qc_dict)    
# print(Q_program.get_qasms(circuits))

# run the program on hardware/simulator
result=Q_program.execute(circuits, backend=backend, shots=shots, wait=20, timeout=600, silent=False)

# arrange the data from the run

keys_0_1=list(result.get_counts('step_0').keys())# get the key of the excited state '00001' 

data=np.zeros(len(qc_dict.keys())) # numpy array for data
sigma_data = np.zeros(len(qc_dict.keys()))

# change unit from ns to microseconds
plot_factor=1
if unit.find('ns')>-1:
    plot_factor=1000
    punit='$\mu$s'
xvals=time_per_step*np.linspace(0,len(qc_dict.keys()),len(qc_dict.keys()))/plot_factor # calculate the time steps in microseconds 

for ii,key in enumerate(qc_dict.keys()):
    # get the data in terms of counts for the excited state normalized to the total number of counts
    data[ii]=float(result.get_counts(key)[keys_0_1[1]])/shots
    sigma_data[ii] = np.sqrt(data[ii]*(1-data[ii]))/np.sqrt(shots)

# fit the data to an exponential    
fitT1, fcov = curve_fit(exp_fit_fun, xvals, data, bounds=([0,2,0], [1., 500, 1])) 
ferr = np.sqrt(np.diag(fcov))

plot_coherence(xvals, data, sigma_data, fitT1, exp_fit_fun, punit, 'T$_1$ ', qubit)

print("a: " + str(round(fitT1[0],2)) + u" \u00B1 " + str(round(ferr[0],2)))
print("T1: " + str(round(fitT1[1],2))+ " µs" + u" \u00B1 " + str(round(ferr[1],2)) + ' µs')
print("c: " + str(round(fitT1[2],2)) + u" \u00B1 " + str(round(ferr[2],2)))

str(params['T1']['value']) +' ' + params['T1']['unit']

# Select qubit to measure T2* on
qubit=0

# Creating registers
qr = Q_program.create_quantum_register('qr', 5)
cr = Q_program.create_classical_register('cr', 5)

params = Q_program.get_backend_parameters(backend)['qubits'][qubit]
pulse_length=params['gateTime']['value'] # single-qubit gate time 
buffer_length=params['buffer']['value'] # spacing between pulses
unit = params['gateTime']['unit']

steps=40
gates_per_step=5
max_gates=(steps-1)*gates_per_step+2

num_osc=4
tot_length=buffer_length+pulse_length
time_per_step=gates_per_step*tot_length
f_set=(4.0/(steps*time_per_step)) # set frequency in MHz
qc_dict={}
for ii in range(steps):
    step_num='step_%s'%(str(ii))
    qc_dict.update({step_num:Q_program.create_circuit(step_num, [qr], [cr])})
    qc_dict[step_num].h(qr[qubit])
    qc_dict[step_num]=pad_QId(qc_dict[step_num],gates_per_step*ii,qr[qubit])
    qc_dict[step_num].u1(2*np.pi*num_osc*ii/(steps-1),qr[qubit])
    qc_dict[step_num].h(qr[qubit])
    qc_dict[step_num].measure(qr[qubit], cr[qubit])
circuits=list(qc_dict)    
# print(Q_program.get_qasms(circuits))

result=Q_program.execute(circuits, backend=backend, shots=shots, wait=20, timeout=600, silent=False)

# arrange the data from the run


keys_0_1=list(result.get_counts('step_0').keys())# get the key of the excited state '00001' 

# change unit from ns to microseconds
plot_factor=1
if unit.find('ns')>-1:
    plot_factor=1000
    punit='$\mu$s'
xvals=time_per_step*np.linspace(0,len(qc_dict.keys()),len(qc_dict.keys()))/plot_factor # calculate the time steps 


data=np.zeros(len(qc_dict.keys())) # numpy array for data
sigma_data = np.zeros(len(qc_dict.keys()))

for ii,key in enumerate(qc_dict.keys()):
    # get the data in terms of counts for the excited state normalized to the total number of counts
    data[ii]=float(result.get_counts(key)[keys_0_1[1]])/shots
    sigma_data[ii] = np.sqrt(data[ii]*(1-data[ii]))/np.sqrt(shots)
    
fitT2s, fcov = curve_fit(osc_fit_fun, xvals, data, p0=[0.5, 100, 1/10, np.pi, 0], bounds=([0.3,0,0,0,0], [0.5, 200, 1/2,2*np.pi,1]))
ferr = np.sqrt(np.diag(fcov))

plot_coherence(xvals, data, sigma_data, fitT2s, osc_fit_fun, punit, '$T_2^*$ ', qubit)

print("a: " + str(round(fitT2s[0],2)) + u" \u00B1 " + str(round(ferr[0],2)))
print("T2*: " + str(round(fitT2s[1],2))+ " µs"+ u" \u00B1 " + str(round(ferr[1],2)) + ' µs')
print("f: " + str(round(10**3*fitT2s[2],3)) + 'kHz' + u" \u00B1 " + str(round(10**6*ferr[2],3)) + 'kHz')
print("phi: " + str(round(fitT2s[3],2)) + u" \u00B1 " + str(round(ferr[3],2)))
print("c: " + str(round(fitT2s[4],2)) + u" \u00B1 " + str(round(ferr[4],2)))

# Select qubit to measure T2 echo on
qubit=3

# Creating registers
qr = Q_program.create_quantum_register('qr', 5)
cr = Q_program.create_classical_register('cr', 5)

params = Q_program.get_backend_parameters(backend)['qubits'][qubit]
pulse_length=params['gateTime']['value'] # single-qubit gate time 
buffer_length=params['buffer']['value'] # spacing between pulses
unit = params['gateTime']['unit']

steps=20
gates_per_step=20
tot_length=buffer_length+pulse_length
max_gates=(steps-1)*2*gates_per_step+3
time_per_step=(2*gates_per_step)*tot_length
qc_dict={}
for ii in range(steps):
    step_num='step_%s'%(str(ii))
    qc_dict.update({step_num:Q_program.create_circuit(step_num, [qr], [cr])})
    qc_dict[step_num].h(qr[qubit])
    qc_dict[step_num]=pad_QId(qc_dict[step_num],gates_per_step*ii,qr[qubit])
    qc_dict[step_num].x(qr[qubit])
    qc_dict[step_num]=pad_QId(qc_dict[step_num],gates_per_step*ii,qr[qubit])
    qc_dict[step_num].h(qr[qubit])
    qc_dict[step_num].measure(qr[qubit], cr[qubit])
circuits=list(qc_dict)    
# print(Q_program.get_qasms(circuits))

# run the program on hardware/simulator
result=Q_program.execute(circuits, backend=backend, shots=shots, wait=20, timeout=600, silent=False)

# arrange the data from the run

keys_0_1=list(result.get_counts('step_0').keys())# get the key of the excited state '00001' 


# change unit from ns to microseconds
plot_factor=1
if unit.find('ns')>-1:
    plot_factor=1000
    punit='$\mu$s'
xvals=time_per_step*np.linspace(0,len(qc_dict.keys()),len(qc_dict.keys()))/plot_factor # calculate the time steps 


data=np.zeros(len(qc_dict.keys())) # numpy array for data
sigma_data = np.zeros(len(qc_dict.keys()))


for ii,key in enumerate(qc_dict.keys()):
    # get the data in terms of counts for the excited state normalized to the total number of counts
    data[ii]=float(result.get_counts(key)[keys_0_1[1]])/shots
    sigma_data[ii] = np.sqrt(data[ii]*(1-data[ii]))/np.sqrt(shots)
    
fitT2e, fcov = curve_fit(exp_fit_fun, xvals, data, bounds=([-1,10,0], [0, 150, 1])) 
ferr = np.sqrt(np.diag(fcov))

plot_coherence(xvals, data, sigma_data, fitT2e, exp_fit_fun, punit, '$T_{2echo}$ ', qubit)

print("a: " + str(round(fitT2e[0],2)) + u" \u00B1 " + str(round(ferr[0],2)))
print("T2: " + str(round(fitT2e[1],2))+ ' µs' + u" \u00B1 " + str(round(ferr[1],2)) + ' µs')
print("c: " + str(round(fitT2e[2],2)) + u" \u00B1 " + str(round(ferr[2],2)))

str(params['T2']['value']) +' ' + params['T2']['unit']

# Select qubit to measure T2 echo on
qubit=3

# Creating registers
qr = Q_program.create_quantum_register('qr', 5)
cr = Q_program.create_classical_register('cr', 5)

params = Q_program.get_backend_parameters(backend)['qubits'][qubit]
pulse_length=params['gateTime']['value'] # single-qubit gate time 
buffer_length=params['buffer']['value'] # spacing between pulses
unit = params['gateTime']['unit']

steps=17
gates_per_step=6
num_echo=7 # has to be odd number to end up in ground state at the end
tot_length=buffer_length+pulse_length
time_per_step=((num_echo+1)*gates_per_step+num_echo)*tot_length
max_gates=num_echo*(steps-1)*gates_per_step+num_echo+2
qc_dict={}
for ii in range(steps):
    step_num='step_%s'%(str(ii))
    qc_dict.update({step_num:Q_program.create_circuit(step_num, [qr], [cr])})
    qc_dict[step_num].h(qr[qubit])
    for iii in range(num_echo):
        qc_dict[step_num]=pad_QId(qc_dict[step_num], gates_per_step*ii, qr[qubit])
        qc_dict[step_num].x(qr[qubit])
    qc_dict[step_num]=pad_QId(qc_dict[step_num], gates_per_step*ii, qr[qubit])
    qc_dict[step_num].h(qr[qubit])
    qc_dict[step_num].measure(qr[qubit], cr[qubit])
circuits=list(qc_dict)    
# print(Q_program.get_qasms(circuits))

# run the program on hardware/simulator
result=Q_program.execute(circuits, backend=backend, shots=shots, wait=20, timeout=600, silent=False)

# arrange the data from the run

keys_0_1=list(result.get_counts('step_0').keys())# get the key of the excited state '00001' 


# change unit from ns to microseconds
plot_factor=1
if unit.find('ns')>-1:
    plot_factor=1000
    punit='$\mu$s'
xvals=time_per_step*np.linspace(0,len(qc_dict.keys()),len(qc_dict.keys()))/plot_factor # calculate the time steps 


data=np.zeros(len(qc_dict.keys())) # numpy array for data
sigma_data = np.zeros(len(qc_dict.keys()))

for ii,key in enumerate(qc_dict.keys()):
    # get the data in terms of counts for the excited state normalized to the total number of counts
    data[ii]=float(result.get_counts(key)[keys_0_1[1]])/shots
    sigma_data[ii] = np.sqrt(data[ii]*(1-data[ii]))/np.sqrt(shots)
    
fitT2cpmg, fcov = curve_fit(exp_fit_fun, xvals, data, bounds=([-1,10,0], [0, 150, 1])) 
ferr = np.sqrt(np.diag(fcov))

plot_coherence(xvals, data, sigma_data, fitT2cpmg, exp_fit_fun, punit, '$T_{2cpmg}$ ', qubit)

print("a: " + str(round(fitT2cpmg[0],2)) + u" \u00B1 " + str(round(ferr[0],2)))
print("T2: " + str(round(fitT2cpmg[1],2))+ ' µs' + u" \u00B1 " + str(round(ferr[1],2)) + ' µs')
print("c: " + str(round(fitT2cpmg[2],2)) + u" \u00B1 " + str(round(ferr[2],2)))

get_ipython().run_line_magic('run', '"../version.ipynb"')



