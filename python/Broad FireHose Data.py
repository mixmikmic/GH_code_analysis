import pandas as pd
import gzip
import os
mut_data = pd.read_csv('../data/muts.txt', sep='\t')
mut_data.columns

data_path = '../data/broadFireHose'
dataset_list = list(filter(lambda x: x.startswith('gdac') and not x.endswith('tar.gz'), os.listdir(data_path)))

file_list = list(
    filter(lambda x: x.endswith('.maf.txt'),
           os.listdir(os.path.join(data_path, dataset_list[0]))
          ))

import re
go_cols = ['GO_Biological_Process', 'GO_Cellular_Component', 'GO_Molecular_Function']
selected_cols = ["Hugo_Symbol", 
    "Entrez_Gene_Id", "NCBI_Build", "Chromosome", 
    "Start_position", "End_position","Strand", 
    "Variant_Classification", "Variant_Type", 
    "Reference_Allele", "Tumor_Seq_Allele2"
] + go_cols

all_data = pd.DataFrame()
for ds in dataset_list: 
    file_list = list(
        filter(lambda x: x.endswith('.maf.txt'),
               os.listdir(os.path.join(data_path, ds))
              ))
    
    for fs in file_list: 
        cohort = re.match( r'^gdac.broadinstitute.org_(.*).Mutation_Packager(.*)', ds).group(1)
        patient =  fs.split('.')[0]
        print("adding {}, {}, currently at: {}".format(cohort, patient, len(all_data)))
        try: 
            data = pd.read_csv(os.path.join(data_path, ds, fs), skiprows=3, sep="\t")
            output = data[selected_cols]
            output['patient'] = patient
            output['type'] = cohort
            all_data = pd.concat([all_data, output[['patient', 'type'] + selected_cols]], axis=0)
        except: 
            pass

all_data.to_csv('../data/muts_eight_cohorts_oncotated.txt', index=False)



