ob_solve_json = """ 
{
  "ob_atom": {
    "decays": [
      { "channels": [[0,1], [1,2]], 
        "rate": 1.0
      }
    ],
    "energies": [],
    "fields": [
      {
        "coupled_levels": [
          [0, 1]
        ],
        "detuning": 0.0,
        "detuning_positive": true,
        "label": "probe",
        "rabi_freq": 5.0,
        "rabi_freq_t_args": {},
        "rabi_freq_t_func": null
      },
      {
        "coupled_levels": [
          [1, 2]
        ],
        "detuning": 0.0,
        "detuning_positive": false,
        "label": "coupling",
        "rabi_freq": 10.0,
        "rabi_freq_t_args": {},
        "rabi_freq_t_func": null
      }
    ],
    "num_states": 3
  },
  "t_min": 0.0,
  "t_max": 1.0,
  "t_steps": 100,
  "method": "mesolve",
  "opts": {}
} """

from maxwellbloch import ob_solve

ob_lamda_solve = ob_solve.OBSolve().from_json_str(ob_solve_json)
ob_lamda_solve.solve(show_pbar=True);

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

pop_0 = np.absolute(ob_lamda_solve.ob_atom.states_t()[:,0,0]) # Ground state population
pop_1 = np.absolute(ob_lamda_solve.ob_atom.states_t()[:,1,1]) # Excited state population
pop_2 = np.absolute(ob_lamda_solve.ob_atom.states_t()[:,2,2]) # Excited state population

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(ob_lamda_solve.tlist, pop_0, label='Ground state')
ax.plot(ob_lamda_solve.tlist, pop_1, label='Excited state')
ax.plot(ob_lamda_solve.tlist, pop_2, label='Other ground state')
ax.set_xlabel(r'Time ($\mu s$)')
ax.set_ylabel(r'Population')
ax.set_ylim([0.,1])
ax.legend(frameon=True)

plt.savefig('images/ob-solve-lamda-on-resonance.png')

