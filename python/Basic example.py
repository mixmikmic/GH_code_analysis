get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir('..')
import prosumpy as pros

demand = pd.Series.from_csv('./tests/data/demand_example.csv')
pv_1kW = pd.Series.from_csv('./tests/data/pv_example.csv')

pv_size = 10

param_tech = {'BatteryCapacity': 30,
              'BatteryEfficiency': .9,
              'InverterEfficiency': 0.96,
              'timestep': .25,
              'MaxPower': 20
             }

pv = pv_1kW * pv_size

E = pros.dispatch_max_sc(pv, demand, param_tech, return_series=False)

E.keys()

pros.print_analysis(pv, demand, param_tech, E)

pros.plot_dispatch(pv, demand, E, week=30)

