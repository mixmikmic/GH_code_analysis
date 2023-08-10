import bioalerts
import numpy as np

from bioalerts import LoadMolecules, Alerts, FPCalculator

from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import time

import sys, numpy as np, scipy as sc, rdkit, matplotlib as pylab, pandas as pd, IPython
print " Python:", sys.version, "\n"
print " Numpy:", np.__version__
print " Scipy:", sc.__version__
print " Rdkit:", rdkit.rdBase.rdkitVersion
print " Matplotlib:", pylab.__version__
print " Pandas:", pd.__version__
print " Ipython:", IPython.__version__

training = bioalerts.LoadMolecules.LoadMolecules("./datasets/training_Ahlberg.smi",delimiter="\t",name_field=None)
                                               
training.ReadMolecules(titleLine=False,smilesColumn=0,nameColumn=1)
print "Total number of input molecules: ", len(training.mols)

test = bioalerts.LoadMolecules.LoadMolecules("./datasets/test_Ahlberg.smi",delimiter="\t",name_field=None)
test.ReadMolecules(titleLine=True)
print "Total number of input molecules: ", len(test.mols)

labels_training = np.genfromtxt('./datasets/training_Ahlberg_activities.txt',
                              dtype='str',
                              skip_header=0,
                              usecols=0)
arr = np.arange(0,len(labels_training))
mask = np.ones(arr.shape,dtype=bool)
mask[training.molserr]=0
labels_training =  labels_training[mask]

labels_test = np.genfromtxt('./datasets/test_Ahlberg_activities.txt',
                              dtype='str',
                              skip_header=0,
                              usecols=0)
arr = np.arange(0,len(labels_test))
mask = np.ones(arr.shape,dtype=bool)
mask[test.molserr]=0
labels_test =  labels_test[mask]

print len(labels_training), len(labels_test)

training_set_info = bioalerts.LoadMolecules.GetDataSetInfo(name_field=None)

initial_time = time.clock()
training_set_info.extract_substructure_information(radii=[1,2,3,4],mols=training.mols)
print round(time.clock() - initial_time)/60, " minutes"

initial_time = time.clock()
# the maximum radius corresponds to the maximum value in the list radii used above (i.e. [2,3,4])
Alerts_categorical_70 = bioalerts.Alerts.CalculatePvaluesCategorical(max_radius=4)


Alerts_categorical_70.calculate_p_values(mols=test.mols,
                                      substructure_dictionary=training_set_info.substructure_dictionary,
                                      bioactivities=labels_training,
                                      mols_ids=training.mols_ids,
                                      threshold_nb_substructures = 5,
                                      threshold_pvalue = 0.05,
                                      threshold_frequency = 0.7,
                                      Bonferroni=False,
                                      active_label='POS',
                                      inactive_label='NEG')
print round(time.clock() - initial_time)/60, " minutes"

initial_time = time.clock()
Alerts_categorical_80 = bioalerts.Alerts.CalculatePvaluesCategorical(max_radius=4)


Alerts_categorical_80.calculate_p_values(mols=test.mols,
                                      substructure_dictionary=training_set_info.substructure_dictionary,
                                      bioactivities=labels_training,
                                      mols_ids=training.mols_ids,
                                      threshold_nb_substructures = 5,
                                      threshold_pvalue = 0.05,
                                      threshold_frequency = 0.8,
                                      Bonferroni=False,
                                      active_label='POS',
                                      inactive_label='NEG')
print round(time.clock() - initial_time)/60, " minutes"

initial_time = time.clock()
Alerts_categorical_80_Bonferroni_True = bioalerts.Alerts.CalculatePvaluesCategorical(max_radius=4)


Alerts_categorical_80_Bonferroni_True.calculate_p_values(mols=test.mols,
                                      substructure_dictionary=training_set_info.substructure_dictionary,
                                      bioactivities=labels_training,
                                      mols_ids=training.mols_ids,
                                      threshold_nb_substructures = 5,
                                      threshold_pvalue = 0.05,
                                      threshold_frequency = 0.8,
                                      Bonferroni=True,
                                      active_label='POS',
                                      inactive_label='NEG')
print round(time.clock() - initial_time)/60, " minutes"

print len(Alerts_categorical_70.output) # Bonferroni False; 
print len(Alerts_categorical_80.output) # Bonferroni False
print len(Alerts_categorical_80_Bonferroni_True.output) # Bonferroni False

