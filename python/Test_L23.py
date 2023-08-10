import sys
get_ipython().magic('matplotlib inline')

cell_file = '../L23_NoHotSpot.cell.nml'
cell_id = 'L23_NoHotSpot'

import warnings
warnings.simplefilter('ignore')

from pyneuroml.analysis import generate_current_vs_frequency_curve
    
    
curve = generate_current_vs_frequency_curve(cell_file, 
                                    cell_id, 
                                    custom_amps_nA =       [-0.11,-0.07,0,0.07,0.11,0.21,0.27,0.35], 
                                    analysis_duration =    1000, 
                                    pre_zero_pulse =       100,
                                    post_zero_pulse =      100,
                                    analysis_delay =       0,
                                    dt =                   0.025,
                                    simulator =            'jNeuroML_NEURON',
                                    plot_voltage_traces =  True,
                                    plot_if =              False,
                                    plot_iv =              False,
                                    temperature =          '35degC',
                                    title_above_plot =      True)

# Longer duration, more points 
    
curve = generate_current_vs_frequency_curve(cell_file, 
                                    cell_id, 
                                    start_amp_nA =         -0.1, 
                                    end_amp_nA =           0.5, 
                                    step_nA =              0.025, 
                                    analysis_duration =    2000, 
                                    pre_zero_pulse =       0,
                                    post_zero_pulse =      0,
                                    analysis_delay =       100,
                                    simulator =            'jNeuroML_NEURON',
                                    plot_voltage_traces =  False,
                                    plot_if =              True,
                                    plot_iv =              True,
                                    temperature =          '35degC',
                                    title_above_plot =      True)

from pyneuroml import pynml
pynml.run_jneuroml("", cell_file, '-png')
from IPython.display import Image
Image(filename=cell_file.replace('.nml','.png'),width=500)



