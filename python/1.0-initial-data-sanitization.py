get_ipython().magic('matplotlib inline')
# standard
import sys
import os

# pandas
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# needed for project imports
sys.path.append(os.path.join(os.getcwd(), "../.."))

# project imports
from housepredictor.extractor import DictMultiExtractor


# this styling is purely my preference
# less chartjunk
sns.set_context('notebook', font_scale=1.5, rc={'line.linewidth': 2.5})
sns.set(style='ticks', palette='Set2')

# read the data 
raw_data = pd.read_json('../data/raw/scrape-results.json')
raw_dict = raw_data.loc[0, 'data']
raw_dict

KEY_NAMES = [
'AangebodenSindsTekst',
'AanmeldDatum',
'AantalKamers',
'AantalKavels',
'Aanvaarding',
'Adres',
'Afstand',
'BronCode',
'DatumOndertekeningAkte',
'GewijzigdDatum',
'GlobalId',
'HeeftOpenhuizenTopper',
'HeeftOverbruggingsgrarantie',
'HeeftTophuis',
'HeeftVeiling',
'InUnitsVanaf',
'IsVerkocht',
'IsVerkochtOfVerhuurd',
'Koopprijs',
'KoopprijsTot',
'Note',
'Oppervlakte',
'Perceeloppervlakte',
'Postcode',
'Prijs.GeenExtraKosten',
'Prijs.OriginelePrijs',
'PromoLabel.HasPromotionLabel',
'PromoLabel.PromotionType',
'PromoLabel.RibbonColor',
'PublicatieDatum',
'PublicatieStatus',
'Soort-aanbod',
'SoortAanbod',
'StartOplevering',
'WGS84_X',
'WGS84_Y',
'WoonOppervlakteTot',
'Woonoppervlakte',
'AangebodenSinds',
'AantalBadkamers',
'AantalSlaapkamers',
'AantalWoonlagen',
'AfgekochtDatum',
'BalkonDakterras',
'BijdrageVVE',
'Bijzonderheden',
'Bouwjaar',
'Bouwvorm',
'EigendomsSituatie',
'Energielabel.Definitief',
'Energielabel.Index',
'Energielabel.Label',
'Energielabel.NietBeschikbaar',
'Energielabel.NietVerplicht',
'ErfpachtBedrag',
'Garage',
'GarageIsolatie',
'GarageVoorzieningen',
'GelegenOp',
'HoofdTuinType',
'IndBasisPlaatsing',
'Inhoud',
'IsIngetrokken',
'Isolatie',
'Ligging',
'ObjectType',
'ObjectTypeMetVoorvoegsel',
'PerceelOppervlakte',
'PermanenteBewoning',
'SchuurBerging',
'SchuurBergingIsolatie',
'SchuurBergingVoorzieningen',
'ServiceKosten',
'SoortDak',
'SoortGarage',
'SoortParkeergelegenheid',
'SoortPlaatsing',
'SoortWoning',
'ToonBezichtigingMaken',
'ToonBrochureAanvraag',
'ToonMakelaarWoningaanbod',
'ToonReageren',
'TuinLigging',
'Verwarming',
'VolledigeOmschrijving',
'Voorzieningen',
'WarmWater',
'WoonOppervlakte',
'WoonOppervlakteTot',
'KoopPrijs',
]

def extract_preliminary(data):
    extraction_specs = [{'key': key} for key in KEY_NAMES]
    extractor = DictMultiExtractor(extraction_specs, sep='.')
    return pd.DataFrame(data.apply(extractor).tolist())
    

data = extract_preliminary(raw_data['data'])

def duplicate_columns(df, return_dataframe = False, verbose = False):
    '''
        a function to detect and possibly remove duplicated columns for a pandas dataframe
    '''
    from pandas.core.common import array_equivalent
    # group columns by dtypes, only the columns of the same dtypes can be duplicate of each other
    groups = df.columns.to_series().groupby(df.dtypes).groups
    duplicated_columns = []

    for dtype, col_names in groups.items():
        column_values = df[col_names]
        num_columns = len(col_names)

        # find duplicated columns by checking pairs of columns, store first column name if duplicate exist 
        for i in range(num_columns):
            column_i = column_values.iloc[:,i].values
            for j in range(i + 1, num_columns):
                column_j = column_values.iloc[:,j].values
                if array_equivalent(column_i, column_j):
                    if verbose: 
                        print("column {} is a duplicate of column {}".format(col_names[i], col_names[j]))
                    duplicated_columns.append(col_names[i])
                    break
    if not return_dataframe:
        # return the column names of those duplicated exists
        return duplicated_columns
    else:
        # return a dataframe with duplicated columns dropped 
        return df.drop(labels = duplicated_columns, axis = 1)

duplicate_col_names = duplicate_columns(data)
deduplicated_data = duplicate_columns(data, return_dataframe=True)
data = deduplicated_data

non_num_cols = data.dtypes[data.dtypes == object].index.tolist()
str_cols = data[non_num_cols]
print('\n'.join(non_num_cols))
str_cols.head()

data = deduplicated_data
good_cols = [col_name for col_name in data.columns if data[col_name].unique().size >= 2]
good_cols





