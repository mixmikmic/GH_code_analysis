# import biothing explorer (current in local, will make it an independent python package)
from visual import pathViewer
# pathViewer is a Python class for graphically display API connection maps and explore bio-entity relaitonships
k = pathViewer()
#Show How APIs/Endpoints/Bio-Entities can be connected together
# set display_graph=True to display the api road map
k.show_api_road_map(display_graph=False)

# this command finds the API Endpoints which connect from the start position (Drug Name) to the end position (SNOMED CT ID)
k.find_path(start='Drug Name', end='SNOMED CT', display_graph=False)

# This command feed drug name 'riluzole' to the path selected, and get 'snomed id' as the output
k.find_output(path=k.paths[0], value='riluzole', display_graph=False)

# This command summarize the result
k.result_summary()

# This command printout the results, which are 'SNOMED CT IDs' that are indications for drug 'riluzole'
k.final_results

# This part reads in the DOID_SNOMED conversion file as a data frame
import pandas as pd
file_url = 'https://raw.githubusercontent.com/NCATS-Tangerine/cq-notebooks/master/Orange_QB2_Other_CQs/Drug_Repurpose_By_Pheno/doid_xref.csv'
df = pd.read_csv(file_url,names=['structure_id', 'doid', 'xref', 'xref_id'])
# This part converts snomed id found above to doid
doid_list = []
for snomed_id in k.final_results['riluzole']:
    doid = df.loc[df['xref_id'] == snomed_id]['doid']
    for _doid in doid:
        doid_list.append(_doid)
print(doid_list)

# this command finds the API Endpoints which connect from the start position (DOID) to the end position (HPO ID)
k.find_path(start='Human Disease Ontology', end='Human Phenotype Ontology', display_graph=False)

# This command feed DOID List ['DOID:332', 'DOID:1227', 'DOID:3082', 'DOID:2741] to the path selected, and get 'HPO ID' as the output
k.find_output(path=k.paths[0], value=doid_list, display_graph=False)

# This command summarize the result
k.result_summary()

# This command printout the results, which are 'HPO IDs' that are phenotypes for the DOIDs list
# only print out the first 30 phenotypes for DOID:332 here.
HPO_ID_332 = ['HP:' + _result for _result in k.final_results['DOID:332']]
HPO_ID_332[0:30]

