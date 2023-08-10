import escher
import cobra
import cameo

escher.list_available_maps()

b = escher.Builder(map_name='e_coli_core.Core metabolism')
b.display_in_notebook()

model = cobra.io.read_sbml_model('data/e_coli_core.xml.gz')
solution = model.optimize()
print('Growth rate: %.2f' % solution.objective_value)

b = escher.Builder(map_name='e_coli_core.Core metabolism',
                   reaction_data=dict(solution.fluxes),
                   # change the default colors
                   reaction_scale=[{'type': 'min', 'color': '#cccccc', 'size': 4},
                                   {'type': 'value', 'value': 0.1, 'color': '#cccccc', 'size': 8},
                                   {'type': 'mean', 'color': '#0000dd', 'size': 20},
                                   {'type': 'max', 'color': '#ff0000', 'size': 40}],
                   # absolute value and no text for data
                   reaction_styles=['size', 'color', 'abs'],
                   # only show the primary metabolites
                   hide_secondary_metabolites=True)
b.display_in_notebook()

import pandas as pd
metabolomics = pd.read_table('data/S4_McCloskey2013_aerobic_metabolomics.csv', sep=',', header=None)
metabolomics.head()

metabolomics_dict = dict(metabolomics.values)

b = escher.Builder(map_name='e_coli_core.Core metabolism',
                   metabolite_data=metabolomics_dict,
                   metabolite_scale=[
                       {'type': 'min', 'color': 'white', 'size': 10},
                       {'type': 'median', 'color': 'green', 'size': 20},
                       {'type': 'max', 'color': 'red', 'size': 40},
                   ],
                   enable_tooltips=False, 
                  )
b.display_in_notebook()

b = escher.Builder(map_name='e_coli_core.Core metabolism',
                   show_gene_reaction_rules=True,
                  )
b.display_in_notebook()

rnaseq = pd.read_table('data/S6_RNA-seq_aerobic_to_anaerobic.csv', sep=',', header=0, index_col=0)
rnaseq.head()

rnaseq_array = [dict(zip(rnaseq.index, x)) for x in rnaseq.values.T]

b = escher.Builder(map_name='e_coli_core.Core metabolism',
                   gene_data=rnaseq_array,
                   reaction_compare_style='log2_fold',
                   # change the default colors
                   reaction_scale=[{'type': 'min', 'color': 'green', 'size': 25},
                                   {'type': 'value', 'value': 0, 'color': '#cccccc', 'size': 8},
                                   {'type': 'max', 'color': 'red', 'size': 25}],
                   # absolute value and no text for data
                   reaction_styles=['size', 'color', 'text'],
                   # only show the primary metabolites
                   hide_secondary_metabolites=True)
b.display_in_notebook()

# pass the model to a new builder
b = escher.Builder(map_json='data/custom_map.json')
b.display_in_notebook()

b = escher.Builder(map_json='data/CHANGE_ME.json')
b.display_in_notebook()

