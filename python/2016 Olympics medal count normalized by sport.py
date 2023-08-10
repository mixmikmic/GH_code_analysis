# Default imports
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# Load data
df = pd.read_csv('now_medals.csv')
print df[:5]

# Identify all sports names
sports_names = df.sport.unique()
print len(sports_names)
print sports_names

# Identify all medal names
medal_names = df.medal.unique()
print len(medal_names)
print medal_names

# Identify all countries
country_names = df.country.unique()
print len(country_names)
print country_names

# Calculate number of medals for each sport
Nmedals_sport = {}
for s in sports_names:
    Nmedals_sport[s] = {}
    for m in medal_names:
        Nmedals = df[df.sport==s][df.medal==m].N.sum()
        Nmedals_sport[s][m] = Nmedals
    Nmedals_sport[s]['total'] = np.sum(Nmedals_sport[s].values())

# Calculate number of medals per country per sport
df_Nmedals = pd.DataFrame(index=country_names,columns=sports_names)
for s in sports_names:
    for c in country_names:
        df_Nmedals.loc[c][s] = df[df.sport==s][df.country==c].N.sum()
print df_Nmedals.loc['United States']['swimming']

df_Nmedalsnorm = pd.DataFrame(index=country_names,columns=sports_names)
for s in sports_names:
    df_Nmedalsnorm[s] = df_Nmedals[s] / Nmedals_sport[s]['total']
print df_Nmedalsnorm.loc['United States']['swimming']

# Calculate number of gold medals per country per sport
df_Ngolds = pd.DataFrame(index=country_names,columns=sports_names)
for s in sports_names:
    for c in country_names:
        df_Ngolds.loc[c][s] = df[df.sport==s][df.country==c][df.medal=='gold'].N.values[0]
print df_Ngolds.loc['United States']['swimming']

df_Ngoldsnorm = pd.DataFrame(index=country_names,columns=sports_names)
for s in sports_names:
    if Nmedals_sport[s]['gold'] == 0:
        df_Ngoldsnorm[s] = 0
    else:
        df_Ngoldsnorm[s] = df_Ngolds[s] / Nmedals_sport[s]['gold']
print df_Ngoldsnorm.loc['United States']['swimming']

# Calculate number of weighted medals per country per sport
df_NmedalsW = pd.DataFrame(index=country_names,columns=sports_names)
for s in sports_names:
    for c in country_names:
        df_NmedalsW.loc[c][s] = df[df.sport==s][df.country==c][df.medal=='gold'].N.values[0]*3 +                                 df[df.sport==s][df.country==c][df.medal=='silver'].N.values[0]*2 +                                 df[df.sport==s][df.country==c][df.medal=='bronze'].N.values[0]
print df_NmedalsW.loc['United States']['swimming']

df_NmedalsWnorm = pd.DataFrame(index=country_names,columns=sports_names)
for s in sports_names:
    df_NmedalsWnorm[s] = df_NmedalsW[s] / (3*Nmedals_sport[s]['gold']+2*Nmedals_sport[s]['silver']+Nmedals_sport[s]['bronze'])
print df_NmedalsWnorm.loc['United States']['swimming']

# Calculate array that's total number of medals for each country
total_medals = []
total_medalsnorm = []
total_medalsW = []
total_medalsWnorm = []
total_golds = []
total_goldsnorm = []
for c in country_names:
    total_medals.append(df_Nmedals.loc[c].sum())
    total_medalsnorm.append(df_Nmedalsnorm.loc[c].sum())
    total_medalsW.append(df_NmedalsW.loc[c].sum())
    total_medalsWnorm.append(df_NmedalsWnorm.loc[c].sum())
    total_golds.append(df_Ngolds.loc[c].sum())
    total_goldsnorm.append(df_Ngoldsnorm.loc[c].sum())

df_Nmedals['total'] = np.array(total_medals)
df_Nmedalsnorm['total'] = np.array(total_medalsnorm)
df_NmedalsW['total'] = np.array(total_medalsW)
df_NmedalsWnorm['total'] = np.array(total_medalsWnorm)
df_Ngolds['total'] = np.array(total_golds)
df_Ngoldsnorm['total'] = np.array(total_goldsnorm)

# Output
df_out1 = pd.DataFrame(index=country_names,columns=['Medals', 'Gold medals', 'Weighted medals', 'Normalized medals', 'Normalized gold medals', 'Normalized weighted medals'])
df_out1['Medals'] = np.array(total_medals)
df_out1['Normalized medals'] = np.array(total_medalsnorm)
df_out1['Weighted medals'] = np.array(total_medalsW)
df_out1['Normalized weighted medals'] = np.array(total_medalsWnorm)
df_out1['Gold medals'] = np.array(total_golds)
df_out1['Normalized gold medals'] = np.array(total_goldsnorm)

df_out1.to_csv('Normalized medals.csv')



