import pandas as pd
# input file "visualize ARO"
df = pd.read_excel('../../visualizeARO.xlsx') # ARO pattern
mic = pd.read_excel('../../data/PATRIC_genomes_AMR.xlsx', sheetname = 'AMR_MATRIX_MIC') # MIC data
ris = pd.read_excel('../../data/PATRIC_genomes_AMR.xlsx', sheetname = 'AMR_MATRIX_SIR') # SIR data

# Mapping to each other
combined = df.set_index('ID').join(ris.set_index('genome_id')) # join by genome ID: SIR<---->ARO
combined = combined.join(mic.set_index('genome_id'), lsuffix = '_ris', rsuffix = '_mic') # join by genome ID: SIR<-->ARO<-->MIC

### CLEAN DATA

# find all-zero columns
count = combined.count(axis = 0, numeric_only = False)
remove = (count == 0)
columnName = list(remove.index)
allZero = list(remove)

# remove them: most are without antibiotics data
print(combined.shape, "before")
for i in range(len(allZero)):
    if allZero[i] == True:
        del combined[columnName[i]]
print(combined.shape, "after")

# there are 407 genes and the remaining are mic/ris data
l = list(combined.count(axis = 0, numeric_only = False))
l.count(750)

# remove genome_name
del combined['genome_name_mic']
del combined['genome_name_ris']

# set SIR to ordered categorical
for i in list(combined.columns.values):
    if type(i) != int:
        if "_ris" in i:
            combined[i] = combined[i].astype('category')
            combined[i].cat.set_categories(['R ','I ','S '], inplace = True)
            combined[i].cat.as_ordered()
            combined[i].cat.reorder_categories(['R ','I ','S '], ordered = True, inplace = True)
            combined[i] = combined[i].cat.rename_categories(['R','I','S']) # remove that fucking space!!! (ANGRY)
            combined[i] = combined[i].cat.add_categories(['cannot determine']) # for furthur self-annotated MIC

combined.head()

combined.to_pickle('../../combined') # save to pickle
combined.to_excel('../../combined_aro_mic.xlsx')

