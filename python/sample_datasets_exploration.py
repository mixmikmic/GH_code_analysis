import pandas as pd

datasets = {}
with open('../../data/pointAs-93710.csv') as fname:
    datasets['point_as'] = pd.read_csv(fname)
with open('../../data/pointBs-93710.csv') as fname:
    datasets['point_bs'] = pd.read_csv(fname)

for df_name, df in datasets.items():
    print('For {} dataset:'.format(df_name))
    print('There are {} unique rows.'.format(len(df.drop_duplicates())))
    print('This dataset has the following columns: {} \n\n'.format(df.columns.values))

print('There are {} unique point Bs addresses.'.format(datasets['point_bs'].Address.nunique()))



