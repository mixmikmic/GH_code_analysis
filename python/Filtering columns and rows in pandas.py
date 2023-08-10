import pandas as pd

df = pd.read_csv('../data/mlb.csv')

df.head()

df.TEAM

df['TEAM']

df[['NAME', 'TEAM']].head()

cols_of_interest = ['TEAM', 'NAME']
df[cols_of_interest].head()

df.TEAM.unique()

lad = df[df['TEAM'] == 'LAD']

lad.head()

millionaires = df[df['SALARY'] >= 1000000]

millionaires.head()

is_a_2_millionaire = df['SALARY'] >= 2000000
two_millionaires = df[is_a_2_millionaire]

two_millionaires.head()

tx = df[df['TEAM'].isin(['TEX', 'HOU'])]

tx.TEAM.unique()

not_tx = df[~df['TEAM'].isin(['TEX', 'HOU'])]

not_tx.TEAM.unique()

catchers_lm = df[(df['POS'] == 'C') & (df['SALARY'] == 535000)]

catchers_lm

