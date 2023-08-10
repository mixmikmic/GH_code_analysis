import numpy as np
import matplotlib.pyplot as plt
import neuron

def add_synapse_stim(seg, time):
    stim = neuron.h.AlphaSynapse(seg)
    stim.onset = time
    stim.gmax = 0.2
    return stim

def calc(tstop=300):

    # setup soma
    soma = neuron.h.Section(name="soma")
    soma.nseg = 1
    soma.diam = 10
    soma.L = 10
    soma.insert("hh")

    # setup dendrite
    ndend = 2
    dends = []
    for i in range(ndend):
        dend = neuron.h.Section()
        dend.nseg = 5
        dend.L = 300
        dend.diam = 0.5
        dend.Ra = 125
        dend.insert("pas")
        dend.connect(soma, 0)
        dends.append(dend)
    
    # add alpha synapse stimulation
    syns = []
    syns.append(add_synapse_stim(dends[0](0.5), 100))
    syns.append(add_synapse_stim(dends[0](0.5), 100))
    syns.append(add_synapse_stim(dends[0](0.5), 200))
    syns.append(add_synapse_stim(dends[1](0.5), 200))

    # setup recorder
    rec_t = neuron.h.Vector()
    rec_t.record(neuron.h._ref_t)
    rec_v_array = []
    rec_v_array.append(neuron.h.Vector())
    rec_v_array[-1].record(soma(0.5)._ref_v)
    for i in range(ndend):
        rec_v_array.append(neuron.h.Vector())
        rec_v_array[-1].record(dends[i](0.5)._ref_v)

    # initialize and run
    neuron.h.finitialize(-65)
    neuron.run(tstop)
    
    # convert recored information to ndarray
    t = np.array(rec_t.as_numpy())
    v_array = []
    for rec_v in rec_v_array:
        v_array.append(np.array(rec_v.as_numpy()))

    return t, v_array

def plot_voltage_array(t, v_array, title=''):
    for v in v_array:
        plt.plot(t, v)
    plt.xlabel("Time [ms]")
    plt.ylabel("Voltage [mV]")
    plt.axis(xmin=0, xmax=max(t), ymin=-80, ymax=50)
    if title:
        plt.title(title)
    plt.show()

t, v_array = calc()
plot_voltage_array(t, v_array[0:1], 'soma')
plot_voltage_array(t, v_array[1:2], 'dend[0]')
plot_voltage_array(t, v_array[2:3], 'dend[1]')



