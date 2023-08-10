# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi
import re

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)

# HIDDEN
state = pd.DataFrame({
    'County': [
        'De Witt County',
        'Lac qui Parle County',
        'Lewis and Clark County',
        'St John the Baptist Parish',
    ],
    'State': [
        'IL',
        'MN',
        'MT',
        'LA',
    ]
})
population = pd.DataFrame({
    'County': [
        'DeWitt  ',
        'Lac Qui Parle',
        'Lewis & Clark',
        'St. John the Baptist',
    ],
    'Population': [
        '16,798',
        '8,067',
        '55,716',
        '43,044',
    ]
})

state

population

john1 = state.loc[3, 'County']
john2 = population.loc[0, 'County']

(john1
 .lower()
 .strip()
 .replace(' parish', '')
 .replace(' county', '')
 .replace('&', 'and')
 .replace('.', '')
 .replace(' ', '')
)

(john1
 .lower()
 .strip()
 .replace(' parish', '')
 .replace(' county', '')
 .replace('&', 'and')
 .replace('.', '')
 .replace(' ', '')
)

def clean_county(county):
    return (county
            .lower()
            .strip()
            .replace(' county', '')
            .replace(' parish', '')
            .replace('&', 'and')
            .replace(' ', '')
            .replace('.', ''))

([clean_county(county) for county in state['County']],
 [clean_county(county) for county in population['County']]
)

state['County']

state['County'].str.lower()

(state['County']
 .str.lower()
 .str.strip()
 .str.replace(' parish', '')
 .str.replace(' county', '')
 .str.replace('&', 'and')
 .str.replace('.', '')
 .str.replace(' ', '')
)

state['County'] = (state['County']
 .str.lower()
 .str.strip()
 .str.replace(' parish', '')
 .str.replace(' county', '')
 .str.replace('&', 'and')
 .str.replace('.', '')
 .str.replace(' ', '')
)

population['County'] = (population['County']
 .str.lower()
 .str.strip()
 .str.replace(' parish', '')
 .str.replace(' county', '')
 .str.replace('&', 'and')
 .str.replace('.', '')
 .str.replace(' ', '')
)

state

population

state.merge(population, on='County')

