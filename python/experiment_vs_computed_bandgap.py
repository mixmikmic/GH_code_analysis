import numpy as np
import pandas as pd

# Set pandas view options
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# filter warnings messages from the notebook
import warnings
warnings.filterwarnings('ignore')

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval

api_key = None # Set your Citrine API key here. If set as an environment variable 'CITRINE_KEY', set it to 'None'
c = CitrineDataRetrieval() # Create an adapter to the Citrine Database.

df = c.get_dataframe(criteria={'data_type': 'EXPERIMENTAL', 'max_results': 100},
                     properties=['Band gap', 'Temperature'],
                    common_fields=['chemicalFormula'])
df.rename(columns={'Band gap': 'Experimental band gap'}, inplace=True) # Rename column

df.head()

get_ipython().run_cell_magic('time', '', 'from pymatgen import MPRester, Composition\nmpr = MPRester() # provide your API key here or add it to pymatgen\n\ndef get_MP_bandgap(formula):\n    """Given a composition, get the band gap energy of the ground-state structure\n    at that composition\n    \n    Args:\n        composition (string) - Chemical formula\n    Returns:\n        (float) Band gap energy of the ground state structure"""\n    # The MPRester requires integer formuals as input\n    reduced_formula = Composition(formula).get_integer_formula_and_factor()[0]\n    struct_lst = mpr.get_data(reduced_formula)\n    \n    # If there is a structure at this composition, return the band gap energy\n    if struct_lst:\n        return sorted(struct_lst, key=lambda e: e[\'energy_per_atom\'])[0][\'band_gap\']\n    \ndf[\'Computed band gap\'] = df[\'chemicalFormula\'].apply(get_MP_bandgap)')

from matminer.figrecipes.plot import PlotlyFig

pf = PlotlyFig(df, x_title='Experimental band gap (eV)', 
               y_title='Computed band gap (ev)',mode='notebook', 
               fontsize=20, ticksize=15)
pf.xy([('Experimental band gap', 'Computed band gap'), ([0, 10], [0, 10])], 
      modes=['markers', 'lines'], lines=[{}, {'color': 'black', 'dash': 'dash'}],
      labels='chemicalFormula', showlegends=False)

df.head()

