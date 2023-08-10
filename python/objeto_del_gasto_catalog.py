# Generate `Objeto de Gasto` catalog

## File ingestion

from pandas import ExcelFile, read_excel, DataFrame
from slugify import slugify

cd /home/loic/repos/mexico

catalog_file = ExcelFile('objeto_del_gasto.catalog.xlsx')

catalog = {}

for sheet in catalog_file.sheet_names:
    name = slugify(sheet, separator='_')
    catalog[name] = catalog_file.parse(sheet).dropna()
    message = 'Loaded sheet {sheet} into "{name}" ({nb} lines)'
    parameters = dict(sheet=sheet, name=name, nb=len(catalog[name]))
    print(message.format(**parameters))
    print('Columns =', list(catalog[name].columns))

catalog['partida_especifica']['PARTIDA_ESPECIFICA'] =  catalog['partida_especifica']['PARTIDA_ESPECIFICA'].astype(int)
especifica_1 = catalog['partida_especifica'].set_index('PARTIDA_ESPECIFICA').sort_index()
print(especifica_1.info())
especifica_1.head(n=5)

catalog['concatenated']['PARTIDA_ESPECIFICA'] =  catalog['concatenated']['PARTIDA_ESPECIFICA'].astype(int)
especifica_2 = catalog['concatenated'][['PARTIDA_ESPECIFICA', 'DESCRIPCION']].set_index('PARTIDA_ESPECIFICA').sort_index()
print(especifica_2.info())
especifica_2.head(n=5)

comparaison = especifica_1 == especifica_2

comparaison.all()

especifica_1.loc[11101]

from os.path import join

def generate_catalog(file):
    
    new_columns = {}
    catalog_file = ExcelFile(file)
    INDEX_COLUMN = 0
    
    for sheet in catalog_file.sheet_names:
        if sheet != 'Concatenated':
            name = slugify(sheet, separator='_')
            output = join('objeto_del_gasto.catalog', name + '.csv')

            df = catalog_file.parse(sheet).dropna()
            index = df.columns[INDEX_COLUMN]

            df[index] =  df[index].astype(int)
            df.set_index(index, inplace=True)
            df.sort_index(inplace=True)
            
            new_columns[name] = df
            df.to_csv(output)
            
            message = 'Loaded sheet {sheet} into "{name}" ({nb} lines)'
            parameters = dict(sheet=sheet, name=name, nb=len(catalog[name]))

            print(message.format(**parameters))
            print('Columns =', list(catalog[name].columns))
            print('Saved to', output)
            
    return new_columns

catalog = generate_catalog('objeto_del_gasto.catalog.xlsx')

catalog['capitulo'].loc[1000]

catalog['partida_especifica'].loc[21101]

from pandas import read_csv
from os import listdir
from os.path import join

def load_catalogs(folder):
    
    catalogs = {}
    files = listdir(folder)
    
    for file in files:
        name = file.split('.')[0]
        print('Loading', name)
        filepath = join(folder, file)
        
        catalogs[name] = read_csv(filepath)
        index_column = catalogs[name].columns[0]
        catalogs[name].set_index(index_column, inplace=True)
    
        print(catalogs[name].info(), '\n')
    
    return catalogs

catalogs = load_catalogs('objeto_del_gasto.catalog')

catalogs['capitulo'].loc[2000]

from pandas import DataFrame
from numpy import nan, int32

ids = catalogs['capitulo'].sample(n=6).reset_index()
del ids['DESCRIPCION']
ids.loc[5] = nan
ids['CAPITULO'] = ids['CAPITULO'].astype(str, inplace=True)
print(ids.info())
ids

str(nan)

df = DataFrame()
df

ids['CAPITULO']

description = catalog['capitulo'].loc[ids['CAPITULO']]['DESCRIPCION'].reset_index()
del description['CAPITULO']
description

from pandas import concat

concat([description, ids], axis=1)

description = catalog['capitulo'].loc[ids['CAPITULO']].reset_index()
description

from pandas import read_csv

merged = read_csv('pipeline.out/iteration-before-holiday/mexican_federal_budget.merged.csv')
merged.head(n=5)

objeto_raw = merged[['Objeto del Gasto',]]
objeto_raw.head(n=5)

len(objeto_raw)

objeto = objeto_raw.dropna()
len(objeto.dropna())

objeto.head(n=5)

objeto['Objeto del Gasto'] = objeto['Objeto del Gasto'].astype(int)

objeto.describe()

objeto.max()

objeto.min()

small = objeto[objeto['Objeto del Gasto'] <= 10000]

small.head(n=30)

len(small)

merged[merged['Objeto del Gasto'] < 10000].groupby(['Ciclo']).count()

mx2016 = merged[merged['Ciclo'] == 2016]

mx2016['Objeto del Gasto'].sample(n=10)

mx2010 = merged[merged['Ciclo'] == 2010]

mx2010['Objeto del Gasto'].value_counts()



