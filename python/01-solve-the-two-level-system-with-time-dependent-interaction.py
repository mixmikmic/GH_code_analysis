field_dict = {'coupled_levels':[[0,1]], 'rabi_freq': 5.0, 'rabi_freq_t_func': 'square_1', 
              'rabi_freq_t_args': { 'ampl_1': 1.0, 'on_1': 0.2, 'off_1': 0.8}} # [2π MHz]

from maxwellbloch import ob_atom
ob_two = ob_atom.OBAtom(num_states=2, fields=[field_dict], decays=[])

import numpy as np
tlist = np.linspace(0., 1., 201) # [µs]

ob_two.mesolve(tlist, show_pbar=True)

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

pop_0 = np.absolute(ob_two.states_t()[:,0])**2 # Ground state population
pop_1 = np.absolute(ob_two.states_t()[:,1])**2 # Excited state population

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tlist, pop_0, label='Excited state')
ax.plot(tlist, pop_1, label='Ground state')
ax.set_xlabel(r'Time ($\mu s$)')
ax.set_ylabel(r'Population')
ax.set_ylim([0.,1])

