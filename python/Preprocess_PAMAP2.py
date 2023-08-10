get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import pandas as pd
# mcfly
from mcfly import tutorial_pamap2, modelgen, find_architecture, storage

# Specify in which directory you want to store the data:
directory_to_extract_to = "/media/sf_VBox_Shared/timeseries/"

header = tutorial_pamap2.get_header()
print(header)

all_activities = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 24, 0]
include_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
print(len(include_activities))
exclude_activities = [n for n in all_activities if n not in include_activities]
print(exclude_activities)

columns_to_use = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
                 'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
                 'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z']
outputdir = "cleaned_12activities_9vars"

get_ipython().magic('pdb 1')
tutorial_pamap2.fetch_and_preprocess(directory_to_extract_to, 
                     columns_to_use=columns_to_use, 
                     output_dir=outputdir, 
                     exclude_activities=exclude_activities,
                    fold=True)

columns_to_use = header[2:]
outputdir = "cleaned_12activities_allvars"
tutorial_pamap2.fetch_and_preprocess(directory_to_extract_to, 
                     columns_to_use=columns_to_use, 
                     output_dir=outputdir, 
                     exclude_activities=exclude_activities,
                                    fold=True)



