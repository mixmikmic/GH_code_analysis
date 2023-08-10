get_ipython().run_cell_magic('bash', '', 'ls MPI-Leipzig/behavioral_data_MPILMBB/phenotype | head')

get_ipython().run_cell_magic('bash', '', '\ncat MPI-Leipzig/behavioral_data_MPILMBB/phenotype/BDI.json | head')

get_ipython().run_cell_magic('bash', '', '\nhead MPI-Leipzig/behavioral_data_MPILMBB/phenotype/BDI.tsv')

#Allow us to import python files in scripts
import sys
sys.path.append('./scripts')

import matplotlib.pyplot as plt
import numpy as np

import find_subjects_behavior_data as fsbd

#Arguments that would normally be passed through the command line call
behavior_files = [
    "MPI-Leipzig/behavioral_data_MPILMBB/phenotype/BDI.tsv",
    "MPI-Leipzig/behavioral_data_MPILMBB/phenotype/HADS.tsv",
    "MPI-Leipzig/behavioral_data_MPILMBB/phenotype/NEO.tsv"
]
behavior_keys = [
    "BDI_summary_sum",
    "HADS-D_summary_sum",
    "NEO_N"
]

#Get data using find_subject_data
subjects, complete_subjects, raw_data, complete_raw_data = fsbd.get_data(behavior_files, behavior_keys)

fsbd.draw_figure(behavior_keys, raw_data, complete_raw_data)
plt.show()

get_ipython().run_line_magic('matplotlib', 'notebook')

#Allow us to import python files in scripts
import sys
sys.path.append('./scripts')

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive 

import find_subjects_behavior_data as fsbd

#Arguments that would normally be passed through the command line call
behavior_files = [
    "MPI-Leipzig/behavioral_data_MPILMBB/phenotype/BDI.tsv",
    "MPI-Leipzig/behavioral_data_MPILMBB/phenotype/HADS.tsv",
    "MPI-Leipzig/behavioral_data_MPILMBB/phenotype/NEO.tsv"
]
behavior_keys = [
    "BDI_summary_sum",
    "HADS-D_summary_sum",
    "NEO_N"
]

#Get data using find_subject_data
subjects, complete_subjects, raw_data, complete_raw_data = fsbd.get_data(behavior_files, behavior_keys)

def draw_figure():
    fsbd.draw_figure(behavior_keys, raw_data, complete_raw_data)
    
interactive_plot = interactive(draw_figure)
interactive_plot



