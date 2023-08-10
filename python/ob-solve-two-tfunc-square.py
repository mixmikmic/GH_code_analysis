ob_solve_json = """ 
{
  "ob_atom": {
    "decays": [],
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
        "rabi_freq_t_args": 
        { 
          "ampl_1": 1.0, 
          "on_1": 0.2, 
          "off_1": 0.8
        },
        "rabi_freq_t_func": "square_1"
      }
    ],
    "num_states": 2
  },
  "t_min": 0.0,
  "t_max": 1.0,
  "t_steps": 100,
  "method": "mesolve",
  "opts": {}
} """

from maxwellbloch import ob_solve

ob_two_solve = ob_solve.OBSolve().from_json_str(ob_solve_json)

ob_two_solve.solve(show_pbar=True);

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

pop_0 = np.absolute(ob_two_solve.states_t()[:,0])**2 # Ground state population
pop_1 = np.absolute(ob_two_solve.states_t()[:,1])**2 # Excited state population

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(ob_two_solve.tlist, pop_0, label='Ground state')
ax.plot(ob_two_solve.tlist, pop_1, label='Excited state')
ax.set_xlabel(r'Time ($\mu s$)')
ax.set_ylabel(r'Population')
ax.set_ylim([0.,1])
leg = ax.legend(frameon=True)

plt.savefig('images/ob-solve-two-tfunc-square.png')

