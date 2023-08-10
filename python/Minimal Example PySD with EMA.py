import sys
sys.path
sys.path.append('..your-path../EMAworkbench-master/src')

from IPython.display import Image
Image(filename='../../models/SD_Fever/SIR_model.png')

get_ipython().magic('matplotlib inline')

from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from ema_workbench.em_framework import (ModelEnsemble, ModelStructureInterface,
                                        ParameterUncertainty,
                                        CategoricalUncertainty,
                                        Outcome)
from ema_workbench.util import ema_logging, save_results

import numpy as np
import pandas as pd
import pysd

PySD_model = pysd.read_vensim('../../models/SD_Fever/SIR_Simple.mdl')

#============================================================================================
#    Class to specify Uncertainties, Outcomes, Initialize, Run Model and collect outcomes
#============================================================================================
class SimplePythonModel(ModelStructureInterface):
    #specify uncertainties
    uncertainties = [
                     ParameterUncertainty((0, 1), "contact_infectivity")

                    ]
    #specify outcomes 
    outcomes = [Outcome("TIME", time=True),
                Outcome('infectious', time=True),
                Outcome('cumulative_cases', time=True),
                ]
    #Statemenet required syntactically but not used
    def model_init(self, policy, kwargs):
        pass
    #Method to run model
    def run_model(self, kwargs):
        contact_infectivity = kwargs['contact_infectivity']

        results = RunSIRModel(contact_infectivity)
        #Collect outcomes (prepare for saving in tar.gz)
        for i, outcome in enumerate(self.outcomes):
            result = results[i]
            self.output[outcome.name] = np.asarray(result)
#============================================================================================
#    The Model itself
#============================================================================================
def RunSIRModel(contact_infectivity):
   
    SD_result = PySD_model.run(params={
                            'contact_infectivity':contact_infectivity,
                                       })
    
    time = SD_result.index.values

    return (time, SD_result['infectious'], SD_result['cumulative_cases'])

if __name__ == '__main__':
    #np.random.seed(150) #set the seed for replication purposes
    ema_logging.log_to_stderr(ema_logging.INFO)
    model = SimplePythonModel(None, 'simpleModel') #instantiate the model
    ensemble = ModelEnsemble() #instantiate an ensemble
    ensemble.parallel = False #set if parallel computing
    ensemble.model_structure = model #set the model on the ensemble
    nr_experiments = 2000   #Set number of experiments
    results = ensemble.perform_experiments(nr_experiments) #run experiments
    save_results(results, r'../../data/EMA_Results/SIR{}_v1.tar.gz'.format(nr_experiments))

import matplotlib.pyplot as plt

from ema_workbench.analysis.plotting import envelopes, lines, kde_over_time, multiple_densities
from ema_workbench.analysis.plotting_util import KDE, plot_boxplots, LINES, ENV_LIN, BOXPLOT, VIOLIN
from ema_workbench.util import load_results

results = load_results(r'../../data/EMA_Results/SIR2000_v1.tar.gz')
experiments, outcomes = results

oois = outcomes.keys()[:-1]
for ooi in oois:
    data_to_sort_by = outcomes[ooi][:,-1]
    indices = np.argsort(data_to_sort_by)
    indices = indices[1:indices.shape[0]:50]  
    
    lines(results, outcomes_to_show=ooi, density=KDE, show_envelope=True, experiments_to_show=indices)

plt.show()



