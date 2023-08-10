import pandas as pd

df = pd.read_csv('../../data/imf_indicators.tsv', delimiter='\t', encoding='ISO-8859-1')

df.head()

df.dtypes

df = pd.read_csv('../../data/imf_indicators.tsv', delimiter='\t', thousands=',', encoding='ISO-8859-1')

df.dtypes

df['2017'].value_counts()

df = pd.read_csv('../../data/imf_indicators.tsv', delimiter='\t', thousands=',', na_values=['n/a'], 
                 encoding='ISO-8859-1')

df.dtypes

df['Country'].value_counts()

df[df['Country'] == 'Germany']

df[(df['Country'] == 'Germany') & (df['Units'].isin(['National currency']))]



