get_ipython().magic('matplotlib inline')
# standard
import sys
import os
import re

# pandas
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fancyimpute


from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, Imputer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from scipy import sparse

# needed for project imports
sys.path.append(os.path.join(os.getcwd(), "../.."))

# project imports
from housepredictor.extractor import extract_preliminary

# this styling is purely my preference
# less chartjunk
sns.set_context('notebook', font_scale=1.5, rc={'line.linewidth': 2.5})
sns.set(style='ticks', palette='Set2')

# relevant keys from the json
KEY_NAMES = [
'AantalBadkamers',
'AantalKamers',
'AantalWoonlagen',
'Aanvaarding',
'Adres',
'AfgekochtDatum',
'BalkonDakterras',
'BijdrageVVE',
'Bijzonderheden',
'Bouwjaar',
'Bouwvorm',
'BronCode',
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
'GlobalId',
'Inhoud',
'Isolatie',
'Koopprijs',
'Ligging',
'PerceelOppervlakte',
'Perceeloppervlakte',
'PermanenteBewoning',
'Postcode',
'PublicatieDatum',
'SchuurBerging',
'SchuurBergingIsolatie',
'SchuurBergingVoorzieningen',
'ServiceKosten',
'Soort-aanbod',
'SoortDak',
'SoortParkeergelegenheid',
'SoortPlaatsing',
'SoortWoning',
'TuinLigging',
'Verwarming',
'VolledigeOmschrijving',
'Voorzieningen',
'WGS84_X',
'WGS84_Y',
'WarmWater',
'WoonOppervlakte',
'Woonoppervlakte'
]

EXCLUDED_COLS = ['GlobalId', 'Koopprijs', 'Adres', 'Postcode']

raw_data = pd.read_json('../data/raw/scrape-results.json')
data = extract_preliminary(raw_data['data'], KEY_NAMES)

# preprocessing data
# extracting dates

def extract_nums(series):
    """Extracts dats from the give format"""
    vals = series.apply(lambda x: re.search(r'[0-9]+',  x)[0] if x else None)
    return vals.astype(float)

def apply_series_transform(func, df, cols, inplace=False):
    """Aplies a per column transformation to the df with the given func.
    The columns are replaced respectively"""
    if not inplace:
        df = df.copy()
    for col in cols:
        df[col] = func(df[col])
    return df

   
# helper
def df_from_prefix(vals, prefix):
    """Creates a dataframe from the given values
    where the columns are numbered prefix0, prefix1
    ... prefixn-1 where n is th enumber of columns"""
    if vals.shape[1] == 1:
        col_names = [prefix]  #if only one do not append number
    else:
        col_names = ['{0}{1}'.format(prefix, ind) 
                     for ind in range(vals.shape[1])]
    return pd.DataFrame(vals, columns=col_names)
    

# transformer
datum_parser = FunctionTransformer(lambda x: apply_series_transform(extract_nums, x, ['PublicatieDatum', 'AfgekochtDatum']),
                                    validate=False)

# add features for missing values so that inputation does not miss them
def mark_nans(df):
    """Add columns marking null values befor imputing"""
    nulls = df.isnull().astype(float)
    
    # suffix them with null
    null_cols = nulls.rename(columns={col: col + '_null' for col in df.columns})
    return null_cols

# test it
mark_nans(data).head()

# rough textual feature extractor
def extract_tf_idf(data, **kwargs):
    """Extract textual features """
    return TfidfVectorizer(**kwargs).fit_transform(data)


date_cols = ['PublicatieDatum', 'AfgekochtDatum']

def extract_dates(df):
    """Extract date columns"""
    cols = ['PublicatieDatum', 'AfgekochtDatum']
    extracted = [extract_nums(df[col]) for col in cols]
    return pd.concat(extracted, axis=1).values

cat_cols = ['AantalWoonlagen', 'Aanvaarding', 'Bouwvorm', 
            'BronCode', 'Energielabel.Label', 'PermanenteBewoning', 
            'SchuurBerging', 'Soort-aanbod', 'TuinLigging']

def extract_int_cat(df):
    # extract int categorical data. simple normalized labels
    int_cat_cols = [LabelEncoder().fit_transform(df[col].fillna('')) for col in cat_cols]
    labels = np.stack(int_cat_cols, axis=1).astype(float)
    return labels

def extract_categorical(df):
    """Extract a """
    num_cats = extract_int_cat(df)  # get the labels with NaNs accordingly
    features = []
    for i in range(num_cats.shape[1]): 
        enc_cats = OneHotEncoder(handle_unknown='ignore').fit_transform(num_cats[:, i][:, np.newaxis])
        enc_cats[num_cats[:, i] == 0, :] = None  # nullify missing features
        features.append(enc_cats)
        
    return sparse.hstack(features)

# columns with a sufficiently small number of unique items so we can categorize them

extract_int_cat(data)
a = extract_categorical(data)

# find textual, non-categorical columns
# declare short and long text columns
# short columns should not be restricted to a max number of features
# while the description should
short_text_cols = ['VolledigeOmschrijving',
 'Voorzieningen',
 'GarageVoorzieningen',
 'SoortDak',
 'GarageIsolatie',
 'Ligging',
 'EigendomsSituatie',
 'SchuurBergingIsolatie',
 'Bijzonderheden',
 'BalkonDakterras',
 'WarmWater',
 'SoortWoning',
 'Verwarming',
 'SoortParkeergelegenheid',
 'Garage',
 'Isolatie',
 'SchuurBergingVoorzieningen']
descr_text_cols = ['VolledigeOmschrijving']

def unprefix_dict(d, pref):
    filtered_dict = {k[len(pref):]:v for k,v in d.items() if k.startswith(pref)}
    return filtered_dict

def extract_textual(df, **kwargs):
    """Extract textual data"""
    short_kwargs = unprefix_dict(kwargs, 'short_')
    descr_kwargs = unprefix_dict(kwargs, 'descr_')
    
    short_text_feats = [extract_tf_idf(df[col].fillna(''), 
                                       **short_kwargs) 
                        for col in short_text_cols]
    descr_text_feats = [extract_tf_idf(df[col].fillna(''), 
                                       **descr_kwargs) 
                        for col in descr_text_cols]
    return sparse.hstack(short_text_feats+descr_text_feats)

# list of numerical(and boolean features)
data.columns[data.dtypes != object]

num_cols =['AantalBadkamers', 'AantalKamers', 'BijdrageVVE',
       'Energielabel.Definitief', 'Energielabel.Index',
       'Energielabel.NietBeschikbaar', 'Energielabel.NietVerplicht',
       'ErfpachtBedrag',  'Inhoud', 
       'PerceelOppervlakte', 'Perceeloppervlakte', 'ServiceKosten',
       'SoortPlaatsing','WoonOppervlakte',
       'Woonoppervlakte']

def extract_features(df, use_text=True, **kwargs):
    nan_features = mark_nans(df).values  # boolean feature for nans
    date_features = extract_dates(df)  # date features, not imputed
    num_features = df[num_cols].astype(float)  # numeric values(some bool, convert em)
    cat_features = extract_categorical(df)  # one-hot categorical features
    features = [nan_features, date_features, num_features, cat_features]
    
    # check whether to use textual features
    if use_text:
        text_kwargs = unprefix_dict(kwargs, 'text_')
        text_features = extract_textual(df, **text_kwargs)
        features.append(text_features)
    
    return sparse.hstack(features)
    
feats = extract_features(data, text_descr_use_idf=True, text_descr_max_features=1000)

imp_feats = Imputer(strategy='median').fit_transform(feats)
(imp_feats.toarray() == None).ravel().sum()  # number of remaining null vals

vectorizer = HashingVectorizer()
tf = vectorizer.fit_transform(data['SchuurBergingVoorzieningen'].fillna(''))

