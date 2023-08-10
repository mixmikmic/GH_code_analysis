import sys
from BEE import *

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

# Makes the figures in the PNG format:
# For more information see %config InlineBackend
get_ipython().magic("config InlineBackend.figure_formats=set([u'png'])")

plt.rcParams['figure.figsize'] = 10, 6

voltages = []
def compare_to_brian():
    global voltages
    voltages = []
    # Initialize the arrays with the simulated values
    # and saves the values for t=0s
    t = []
    spikes_idx = numpy.arange(NofN)
    spikes = []
    tstp = 0.2E-3
    ti=0
#     spks = output_sim(NofN)
    t.append(ti)
#     spikes.append(spikes_idx[spks])
    spikes.append(reads_spikes(NofN))
    voltages.append(output_voltages(NofN).mean())

    for i in range(50):
        # Runs one step and save the result
        updates_sim([], [], [], [], 0, 0)

        # Appends to the output
#         spks = output_sim(NofN)
        ti+=tstp
        t.append(ti)
#         spikes.append(spikes_idx[spks])
        spikes.append(reads_spikes(NofN))
        voltages.append(output_voltages(NofN).mean())
        
    AINP=100

    # Injects some spikes into the liquid!
    exc_inputs = range(50)
    inh_inputs = []
    exc_weights = [AINP*1E-9]*50
    inh_weights = []
    updates_sim(exc_inputs, inh_inputs, exc_weights, inh_weights,50,0)

    # Appends to the output
#     spks = output_sim(NofN)
    ti+=tstp
    t.append(ti)
#     spikes.append(spikes_idx[spks])
    spikes.append(reads_spikes(NofN))
    voltages.append(output_voltages(NofN).mean())
    
    for i in range(number_of_steps-51):
        # Runs one step and save the result
        updates_sim([], [], [], [], 0, 0)
        # Appends to the output
#         spks = output_sim(NofN)
        ti+=tstp
        t.append(ti)
#         spikes.append(spikes_idx[spks])
        spikes.append(reads_spikes(NofN))
        voltages.append(output_voltages(NofN).mean())
    return t,spikes

# Running the simulation using the command line interface:
# simulator_main(["4","rdc2","10","-c", "-p"])

BEE_free()

# NofN=15*30*36;
# NoINHN,NoEXCN,NoINHC,NoEXCC=output_stats()

# seeds=numpy.array(numpy.random.randint(0,10000,5),dtype=numpy.uint32)
seeds = numpy.array([7436,  813, 4518, 9905, 5823],dtype=numpy.uint32)
net_shape = numpy.array([15,3,3])
NofN = net_shape.prod()
if not BEE_initialized():
    get_ipython().magic('time initialize_sim(my_net_shape = net_shape, my_lbd_value = 1.2,                          my_seeds=seeds, SpkLiq_vresets = [13.5E-3,13.5E-3+1E-12])')

BEE_initialized()

get_ipython().run_cell_magic('time', '', 'if BEE_initialized() and (not BEE_connected()):\n    generate_connections()')

BEE_connected()

get_ipython().run_cell_magic('time', '', 'if BEE_initialized() and (not BEE_connected()):\n    process_connections()')

BEE_connected()

NofN,NoINHN,NoEXCN,NoINHC,NoEXCC=output_stats()

# Normally the BEE simulator does NOT updates neurons that have no connections to other neurons.
# The neurons without connections are marked in a special array that can be read from the
# Python function output_connected(NofN).
# Every time the simulator goes through one step it checks if the neurons is marked as connected before
# updating its state variables.

# Disconnects all the neurons (nothing will generate spikes anymore)
# control_connected(numpy.zeros(NofN,dtype=numpy.int32))

# Forces the simulator to calculate the values for all neurons (even the unconnected ones)
# This is important during the tests without connections
control_connected(numpy.ones(NofN,dtype=numpy.int32))

number_of_steps = 500

# %timeit t,spikes = compare_to_brian()
get_ipython().magic('time t,spikes = compare_to_brian()')

print "Total number of spikes:",len(numpy.concatenate(spikes))

x_plot = numpy.array([t[ti] for i,ti in zip(spikes,xrange(len(spikes))) for j in i])
y_plot = [j for i,ti in zip(spikes,xrange(len(spikes))) for j in i]

plt.figure();
plt.plot(x_plot*1000,y_plot,'.', markersize=1);
plt.xlim(t[0]*1000-1,t[-1]*1000+1);
plt.ylim(0,NofN);
plt.title("Pure C Simulator");
plt.show();

plt.figure();
plt.plot(voltages);
# plt.xlim(0,500);
plt.title("Mean value of the membrane voltages");
plt.show();

my_STEP = 0.2
AINP = 100 # Input gains in nA

rndst_noisy = numpy.random.RandomState(seeds[4])

import brian

try:
    brian.reinit() # Necessary when using a python console
except:
    print "Brian REINIT error..."

import numpy # I could use "brian.", because Brian imports numpy, but I prefer not.
import time


Net_shape=net_shape
Number_of_neurons_lsm=NofN
voltages_brian = []

# These make easier to use the Brian objects without the "brian." at the beginning
ms = brian.ms
mV = brian.mV
nA = brian.nA
nF = brian.nF
NeuronGroup = brian.NeuronGroup
SpikeGeneratorGroup = brian.SpikeGeneratorGroup
Synapses = brian.Synapses
SpikeMonitor = brian.SpikeMonitor
network_operation = brian.network_operation 
defaultclock = brian.defaultclock

defaultclock.dt = my_STEP*ms



initial_time = time.time()

print "Initial time (in seconds):",initial_time


#
# Number of Inhibitory neurons - LIQUID - 20% of the total neurons
lsm_indices = numpy.arange(Number_of_neurons_lsm) # Generates a list of indices

#### INPUT FROM MY SIMULATOR!!!
# inhibitory_index_L = output_inh_indices(NoINHN)
# inhibitory_index_L = numpy.arange(NoINHN)
# excitatory_index = output_exc_indices(NoEXCN)
# excitatory_index = numpy.arange(NoINHN,NoEXCN)


# Generate the connections matrix inside the Liquid (Liquid->Liquid) - according to Maass2002
#
print "Liquid->Liquid connections..."


#
# These are the cell (neuron) parameters according to Maass 2002
#
cell_params_lsm = {'cm'          : 30*nF,    # Capacitance of the membrane 
                                             # =>>>> MAASS PAPER DOESN'T MENTION THIS PARAMETER DIRECTLY
                                             #       but the paper mentions a INPUT RESISTANCE OF 1MEGA Ohms and tau_m=RC=30ms, so cm=30nF
                   'i_offset'    : 0.0*nA,   # Offset current - random for each neuron from [14.975nA to 15.025nA] => Masss2002 - see code below
                   'tau_m'       : 30.0*ms,  # Membrane time constant => Maass2002
                   'tau_refrac_E': 3.0*ms,   # Duration of refractory period - 3mS for EXCITATORY => Maass2002
                   'tau_refrac_I': 2.0*ms,   # Duration of refractory period - 2mS for INHIBITORY => Maass2002
                   'tau_syn_E'   : 3.0*ms,   # Decay time of excitatory synaptic current => Maass2002
                   'tau_syn_I'   : 6.0*ms,   # Decay time of inhibitory synaptic current => Maass2002
                   'v_reset'     : 13.5*mV,  # Reset potential after a spike => Maass2002
                   'v_rest'      : 0.0*mV,   # Resting membrane potential => Maass2002
                   'v_thresh'    : 15.0*mV,  # Spike threshold => Maass2002
                   'i_noise'     : 0.2*nA    # Used in Maass2002: mean 0 and SD=0.2nA
                }

# IF_curr_exp - MODEL EXPLAINED
# Leaky integrate and fire model with fixed threshold and
# decaying-exponential post-synaptic current. 
# (Separate synaptic currents for excitatory and inhibitory synapses)
lsm_neuron_eqs='''
  dv/dt  = (ie + ii + i_offset + i_noise)/c_m + (v_rest-v)/tau_m : mV
  die/dt = -ie/tau_syn_E                : nA
  dii/dt = -ii/tau_syn_I                : nA
  tau_syn_E                             : ms
  tau_syn_I                             : ms
  tau_m                                 : ms
  c_m                                   : nF
  v_rest                                : mV
  i_offset                              : nA
  i_noise                               : nA
  '''
# lsm_neuron_eqs='''
#   dv/dt  = (ie + ii + i_offset + i_noise)/c_m + (v_rest-v)/tau_m : mV
#   ie                                    : nA
#   ii                                    : nA
#   tau_m                                 : ms
#   c_m                                   : nF
#   v_rest                                : mV
#   i_offset                              : nA
#   i_noise                               : nA
#   '''



########################################################################################################################
#
# LIQUID - Setup
#
print "LIQUID - Setup..."

#### INPUT FROM MY SIMULATOR!!!
refractory_vector=output_refrac_values(NofN).astype(numpy.double)


# This is the population (neurons) used exclusively to the Liquid (pop_lsm).
# All the neurons receive the same threshold and reset voltages.
pop_lsm = NeuronGroup(Number_of_neurons_lsm, model=lsm_neuron_eqs, 
                                             threshold=cell_params_lsm['v_thresh'], 
                                             reset=cell_params_lsm['v_reset'], 
                                             refractory=refractory_vector, 
                                             max_refractory=max(cell_params_lsm['tau_refrac_E'], 
                                                                cell_params_lsm['tau_refrac_I']))


# Here I'm mixing numpy.fill with the access of the state variable "c_m" in Brian (because Brian is using a numpy.array)
# Sets the value of the capacitance according to the cell_params_lsm (same value to all the neurons)
pop_lsm.c_m.fill(cell_params_lsm['cm'])


# Sets the value of the time constant RC (or membrane constant) according to the cell_params_lsm (same value to all the neurons)
pop_lsm.tau_m.fill(cell_params_lsm['tau_m'])

# Sets the i_offset according to Maass2002
# The i_offset current is random, but never changes during the simulation.
# this current should be drawn from a uniform distr [14.975,15.025]
# Joshi2005 does [13.5,14.5] ???? Maybe is to avoid spikes without inputs...
#
#### INPUT FROM MY SIMULATOR!!!
pop_lsm.i_offset=output_noisy_offsets(NofN)*1E9*nA #1E9 is used to convert from A to nA



pop_lsm.tau_syn_E.fill(cell_params_lsm['tau_syn_E']) # (same value to all the neurons)
pop_lsm.tau_syn_I.fill(cell_params_lsm['tau_syn_I']) # (same value to all the neurons)


# All neurons receive the same value to the resting voltage.
pop_lsm.v_rest.fill(cell_params_lsm['v_rest']) # (same value to all the neurons)


# Sets the initial membrane voltage according to Maass2002. Doesn't change during the simulation.
# this current should be drawn from a uniform distr [13.5mV,15.0mV]
#
#### INPUT FROM MY SIMULATOR!!!
pop_lsm.v=output_initial_voltages(NofN)*1E3*mV #1E3 is used to convert from V to mV


#
# Loading or creating the Synapses objects used within the Liquid
print "Liquid->Liquid connections..."

####THE WEIGHTS MUST BE IN nA, not in A!!!!!!
def create_synapses(population, synapse_type, indices_pre, indices_pos, weights):
    #
    # Creates the connections (Synapse object) among the neurons in the liquid
    #
    # synapse_type = 'exc'
    # EXCITATORY (PRE) TO ANYTHING (POST) 
    # synapse_type = 'inh'
    # INHIBITORY (PRE) TO ANYTHING (POST)
    if synapse_type=='exc':
            model_eq='''w : 1'''
            pre_eq='''ie+=w'''

    elif synapse_type=='inh':
            model_eq='''w : 1'''
            pre_eq='''ii+=w'''

    syn_lsm=brian.Synapses(population, population, model=model_eq, pre=pre_eq)
    # Caution about creation of synapses:
    # 1) there is no deletion
    # 2) synapses are added, not replaced (e.g. S[1,2]=True;S[1,2]=True creates 2 synapses)
    for ipre,ipos in zip(indices_pre,indices_pos):        
        syn_lsm[ipre,ipos] = True # here is where the synapse is really created

####THE WEIGHTS MUST BE IN nA, not in A!!!!!!
    syn_lsm.w=numpy.array(weights)*nA

    syn_lsm.delay=0*ms
    
    return syn_lsm


#### INPUT FROM MY SIMULATOR!!!
# Generates the Liquid->Liquid - EXCITATORY synapses
syn_lsm_exc = create_synapses(pop_lsm,'exc', output_pre_e_connections(NoEXCC), output_pos_e_connections(NoEXCC), (output_pre_e_weights(NoEXCC)*1E9).astype(numpy.double))
# the multiplication by *1E9 changes  the weights from A to nA

#### INPUT FROM MY SIMULATOR!!!
# Generates the Liquid->Liquid - INHIBITORY synapses
syn_lsm_inh = create_synapses(pop_lsm,'inh', output_pre_i_connections(NoINHC), output_pos_i_connections(NoINHC), (output_pre_i_weights(NoINHC)*1E9).astype(numpy.double))
# the multiplication by *1E9 changes  the weights from A to nA

print "Liquid->Liquid connections...Done!"

total_number_of_connections_liquid = len(syn_lsm_exc) + len(syn_lsm_inh)

print "Number of inhibitory synapses in the Liquid: " + str(len(syn_lsm_inh)) # DEBUG to verify if it is working
print "Number of excitatory synapses in the Liquid: " + str(len(syn_lsm_exc)) # DEBUG to verify if it is working


# To understand what is being returned:
# pop_lsm: it is necessary to connect the neuron network with the rest of the world
# [syn_lsm_obj, syn_lsm_exc, syn_lsm_inh]: to include these objects at the simulation (net=Net(...); net.run(total_sim_time*ms)); 
# It is a list because is easy to concatenate lists :D

print "LIQUID - Setup...Done!"

#
# End of the LIQUID - Setup
########################################################################################################################


########################################################################################################################
#
# INPUT - Setup
#
print "INPUT - Setup..."

tspk = defaultclock.dt*50 # The neurons spike after 50 time steps!
number_of_spikes = 50

spiketimes = [(i,tspk) for i in range(number_of_spikes)] 
                # The spikes are going to be received during the simulation, 
                # so this is always an empty list when using the step_by_step_brian_sim!


# I'm using only one big input layer because Brian docs say it is better for the performance
SpikeInputs = SpikeGeneratorGroup(number_of_spikes, spiketimes)


#
#
# Here the synapses are created. The synapses created are ALWAYS excitatory because it is 
# connecting through 'ie' in the neuron model!

syn_world_Input = Synapses(SpikeInputs, pop_lsm,
                                     model='''w : 1''',
                                     pre='''ie+=w''')

for i in range(len(spiketimes)):
    syn_world_Input[i,i] = True

weights_input_liquid = numpy.array([AINP]*number_of_spikes)*nA # Creates a numpy.array with number_of_spikes itens

syn_world_Input.w = weights_input_liquid
syn_world_Input.delay=0*ms

print "INPUT - Setup...Done!"

#
# End of the INPUT - Setup (creation of the connections between the Poisson input and the Liquid!)
#
########################################################################################################################


# DON'T FORGET TO INSERT THE FUNCTION "generate_i_noise" INTO THE MONITORS_OBJECT LIST!!!!
# Generates the noisy current at each time step (as seen in Joshi2005)
@network_operation(clock=defaultclock)
def generate_i_noise():
    global voltages_brian
    voltages_brian.append(pop_lsm.v.mean()) #Saves the mean value of the membrane voltages
    # These are the noise currents inside each liquid's neuron
    pop_lsm.i_noise=rndst_noisy.normal(loc=0, scale=cell_params_lsm['i_noise'],size=Number_of_neurons_lsm)*nA
    # Here I'm using a RandomState that was initialized with the SAME seed as my simulator noisy generation!!!

pop_objects = [pop_lsm, SpikeInputs]

syn_objects = [syn_lsm_exc, syn_lsm_inh, syn_world_Input]

monitors_objects = [generate_i_noise]

OutputMonitor=brian.SpikeMonitor(pop_lsm, record=True)
VMonitor=brian.StateMonitor(pop_lsm,'v', record=True)
IEMonitor=brian.StateMonitor(pop_lsm,'ie', record=True)

net = brian.Network(pop_objects + syn_objects + monitors_objects + [OutputMonitor,VMonitor,IEMonitor])

print "Setup time:", time.time()-initial_time
initial_time = time.time()

net.run(number_of_steps*my_STEP*ms)
print "Simulation time:", time.time()-initial_time

plt.figure()
brian.raster_plot(OutputMonitor, markersize=2)
plt.title("Brian's Output")
plt.xlim(0,number_of_steps*my_STEP)
plt.ylim(0,NofN)
plt.show()

plt.figure();
plt.plot(voltages_brian);
plt.show();

fig=plt.figure()
plt.subplot(211)
brian.raster_plot(OutputMonitor,markersize=2)
plt.title("Brian(top) / Simulator (botton)")
plt.xlim(0,number_of_steps*my_STEP)
plt.ylim(0,NofN)
plt.subplot(212)
plt.plot(x_plot*1000,y_plot,'.',markersize=1)
# plt.title("Simulator")
plt.ylabel("Neuron number")
plt.xlabel("Time (ms)")
plt.xlim(0,number_of_steps*my_STEP)
plt.ylim(0,NofN)
plt.show();
fig.subplots_adjust(hspace=4) # Adjust the distance between subplots

fig=plt.figure()
brian.raster_plot(OutputMonitor,markersize=3,color='red')
plt.title("Brian (red) / Simulator (blue)")
plt.plot(x_plot*1000,y_plot,'b.',markersize=1.5)
plt.ylabel("Neuron number")
plt.xlabel("time(ms)")
plt.xlim(0,number_of_steps*my_STEP)
plt.ylim(0,NofN)
plt.show();
fig.subplots_adjust(hspace=4) # Adjust the distance between subplots

fig=plt.figure()
plt.plot(voltages_brian,'g')
plt.title("Mean value of membrane's voltage - Brian(green)/Simulator(blue)")
plt.plot(voltages,'b')
plt.ylabel("Neuron number")
plt.xlabel("time(ms)")
plt.xlim(0,len(voltages_brian))
plt.show();

# Generate the points to the 3D Scatter plot
def show_3d_connections(net_shape, camera=(45,-135), show_what='both', show_discc=False, show_conn=False, show_index=False, show_coord=False, show_arrows=False, markersize=(10,10,70,70), figsize=(100,100)):
    '''
    Prints the neurons and the connections between them.
    net_shape->the same used to generate the liquid (list or tuple)
    show_what='both','exc' or 'inh'->controls which types of neurons should be plotted
    show_discc=False->doesn't show the neurons without connections to other ones
    show_conn=False->shows or not the lines between the connected neurons
    show_number=True->shows the index number of each neuron according to the Neurongroup
    show_coord=True->shows the x,y,z coordinates
    show_arrows=True->shows arrows indicating the direction of connections (very slow!)
    figsize=(width,height)->in millimetres
    markersize=(inh_dot,exc_dot,inh_star,exc_star)->sizes of the markers
    camera=(45,45)->camera angles, in degrees
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    marker_size_inh,marker_size_exc,marker_size_rec_inh,marker_size_rec_exc = markersize
    user_elev,user_azim = camera
    fig_w,fig_h = figsize #figure size in millimetres
    NofN = net_shape[0]*net_shape[1]*net_shape[2]

    # This is how the liquid neurons index are distributed inside the 3D shape
    # Each line is a neuron index and the collumns are its x,y and z positions.
    count_3d=0;
    Neuron3DMatrix = numpy.empty((NofN,3),dtype=numpy.int)
    for zi in range(net_shape[2]):
        for yi in range(net_shape[1]):
            for xi in range(net_shape[0]):
                Neuron3DMatrix[count_3d][0]=xi;
                Neuron3DMatrix[count_3d][1]=yi;
                Neuron3DMatrix[count_3d][2]=zi;
                count_3d+=1;

    _exc_indices = []
    _inh_indices = []
    
    _disconnected = []
    if not show_discc:
        _disconnected = numpy.array(range(NofN))[output_connected(NofN)==0];
    
    if (show_what=='both' or show_what=='inh'):
        _inh_indices = output_inh_indices(NoINHN)
        # Positions of the neurons in the 3D space
        x_inh=[Neuron3DMatrix[i][0] for i in range(NofN) if (i in _inh_indices) and (i not in _disconnected)] 
        y_inh=[Neuron3DMatrix[i][1] for i in range(NofN) if (i in _inh_indices) and (i not in _disconnected)] 
        z_inh=[Neuron3DMatrix[i][2] for i in range(NofN) if (i in _inh_indices) and (i not in _disconnected)] 
        
        recurrent_connections_idx_inh = numpy.array(output_pre_i_connections(NoINHC)[output_pre_i_connections(NoINHC)==output_pos_i_connections(NoINHC)])
        x_rec_inh=[i[0] for i in Neuron3DMatrix[[i for i in recurrent_connections_idx_inh if i not in _disconnected]]] 
        y_rec_inh=[i[1] for i in Neuron3DMatrix[[i for i in recurrent_connections_idx_inh if i not in _disconnected]]]
        z_rec_inh=[i[2] for i in Neuron3DMatrix[[i for i in recurrent_connections_idx_inh if i not in _disconnected]]]

        # List of tuples with (pre synaptic, post synaptic) INHIBITORY->ANYTHING neurons connections indexes
        inh_connect_positions_pre  = [Neuron3DMatrix[i] for i,j in zip(output_pre_i_connections(NoINHC),range(NoINHC)) if  (output_pre_i_connections(NoINHC)[j]!=output_pos_i_connections(NoINHC)[j])]
        inh_connect_positions_post = [Neuron3DMatrix[i] for i,j in zip(output_pos_i_connections(NoINHC),range(NoINHC)) if  (output_pre_i_connections(NoINHC)[j]!=output_pos_i_connections(NoINHC)[j])]        
        
        
    if (show_what=='both' or show_what=='exc'):
        _exc_indices = output_exc_indices(NoEXCN)
        x_exc=[Neuron3DMatrix[i][0] for i in range(NofN) if (i in _exc_indices) and not (i in _disconnected)] 
        y_exc=[Neuron3DMatrix[i][1] for i in range(NofN) if (i in _exc_indices) and not (i in _disconnected)] 
        z_exc=[Neuron3DMatrix[i][2] for i in range(NofN) if (i in _exc_indices) and not (i in _disconnected)] 

        recurrent_connections_idx_exc = numpy.array(output_pre_e_connections(NoEXCC)[output_pre_e_connections(NoEXCC)==output_pos_e_connections(NoEXCC)])
        x_rec_exc=[i[0] for i in Neuron3DMatrix[[i for i in recurrent_connections_idx_exc if i not in _disconnected]]] 
        y_rec_exc=[i[1] for i in Neuron3DMatrix[[i for i in recurrent_connections_idx_exc if i not in _disconnected]]]
        z_rec_exc=[i[2] for i in Neuron3DMatrix[[i for i in recurrent_connections_idx_exc if i not in _disconnected]]]

        # List of tuples with (pre synaptic, post synaptic) EXCITATORY->ANYTHING neurons connections indexes
        exc_connect_positions_pre  = [Neuron3DMatrix[i] for i,j in zip(output_pre_e_connections(NoEXCC),range(NoEXCC)) if (output_pre_e_connections(NoEXCC)[j]!=output_pos_e_connections(NoEXCC)[j])]
        exc_connect_positions_post = [Neuron3DMatrix[i] for i,j in zip(output_pos_e_connections(NoEXCC),range(NoEXCC)) if (output_pre_e_connections(NoEXCC)[j]!=output_pos_e_connections(NoEXCC)[j])]


    fig = plt.figure() # creates the figure for the following plots
    fig.set_size_inches(fig_w/25.4,fig_h/25.4, forward=False) #Set the figure size in inches (1in == 2.54cm)
    ax = fig.add_subplot(111, projection='3d') # setup to only one

    if (show_what=='both' or show_what=='inh'):    
        ax.scatter(x_inh, y_inh, z_inh, c='b', s=[marker_size_inh]*len(x_inh)) # plots the points correnponding to the inhibitory neurons

    if (show_what=='both' or show_what=='exc'):         
        ax.scatter(x_exc, y_exc, z_exc, c='r', s=[marker_size_exc]*len(x_exc)) # plots the points correnponding to the excitatory neurons

    if (show_what=='both' or show_what=='inh'):    
        ax.scatter(x_rec_inh, y_rec_inh, z_rec_inh, c='b', marker='*', s=[marker_size_rec_inh]*len(x_rec_inh)) # plots where a inhibitory neuron has a reccurent connection

    if (show_what=='both' or show_what=='exc'):    
        ax.scatter(x_rec_exc, y_rec_exc, z_rec_exc, c='r', marker='*', s=[marker_size_rec_exc]*len(x_rec_exc)) # plots where a excitatory neuron has a reccurent connection
    
    _what_to_show = []
    if (show_what=='both' or show_what=='exc'):
        _what_to_show += _exc_indices.tolist()
    if (show_what=='both' or show_what=='inh'):
        _what_to_show += _inh_indices.tolist()        
        
    # Insert a label with the position of each neuron according to the positions_list (NeuronGroup)
    if show_index:    
        for t,n in [(Neuron3DMatrix[i],i) for i in range(NofN) if (i in _what_to_show) and (i not in _disconnected)]:
            ax.text(t[0], t[1], t[2], "["+str(n)+"]")

    # Insert a label with the 3D coordinate used to calculate the connection probabilities
    if show_coord:
        for t,n in [(Neuron3DMatrix[i],i) for i in range(NofN) if (i in _what_to_show) and (i not in _disconnected)]:
            ax.text(t[0], t[1], t[2], str(t)+"="+str(n)) # to insert also the coordinates of the point

    #
    # Draw a 3D vector (arrow)
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if show_conn:
        if show_arrows:
            # Plot the EXCITATORY connections
            if (show_what=='both' or show_what=='exc'):    
                for i in range(len(exc_connect_positions_pre)):
                    a = Arrow3D(
                        [ exc_connect_positions_pre[i][0], exc_connect_positions_post[i][0] ], 
                        [ exc_connect_positions_pre[i][1], exc_connect_positions_post[i][1] ], 
                        [ exc_connect_positions_pre[i][2], exc_connect_positions_post[i][2] ], 
                        label='excitatory connections', mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
                    ax.add_artist(a)
        else:
            if (show_what=='both' or show_what=='exc'):    
                # Plot the EXCITATORY connections - FAST VERSION WITHOUT ARROWS
                for i in range(len(exc_connect_positions_pre)):
                    ax.plot(
                        [ exc_connect_positions_pre[i][0], exc_connect_positions_post[i][0] ], 
                        [ exc_connect_positions_pre[i][1], exc_connect_positions_post[i][1] ], 
                        [ exc_connect_positions_pre[i][2], exc_connect_positions_post[i][2] ], 
                        label='excitatory connections', color='#FF0000')

        if show_arrows:
            if (show_what=='both' or show_what=='inh'):    
                # Plot the INHIBITORY connections
                for i in range(len(inh_connect_positions_pre)):
                    a = Arrow3D(
                        [ inh_connect_positions_pre[i][0], inh_connect_positions_post[i][0] ], 
                        [ inh_connect_positions_pre[i][1], inh_connect_positions_post[i][1] ], 
                        [ inh_connect_positions_pre[i][2], inh_connect_positions_post[i][2] ], 
                        label='inhibitory connections', mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
                    ax.add_artist(a)
        else:
            if (show_what=='both' or show_what=='inh'):                
                # Plot the INHIBITORY connections - FAST VERSION WITHOUT ARROWS
                for i in range(len(inh_connect_positions_pre)):
                    ax.plot(
                        [ inh_connect_positions_pre[i][0], inh_connect_positions_post[i][0] ], 
                        [ inh_connect_positions_pre[i][1], inh_connect_positions_post[i][1] ], 
                        [ inh_connect_positions_pre[i][2], inh_connect_positions_post[i][2] ], 
                        label='inhibitory connections', color='#0000FF')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title("Liquid's 3D structure")
    
    ax.view_init(elev=user_elev, azim=user_azim)
    
    plt.show()

# %matplotlib osx
# %matplotlib inline

show_3d_connections(net_shape, camera=(45,-135), show_what='both', show_discc=False, show_conn=True, show_index=False, show_coord=False, show_arrows=False, markersize=(10,10,120,120),figsize=(500,300))

print "Percentage of Disconnected Neurons:", 100*(len(numpy.array(range(NofN))[output_connected(NofN)==0])/float(NofN))



