import pandas as pd
import numpy as np

df = pd.read_csv("../data/2016_ML_contest_training_data.csv")
df.head()

df.describe()

df = df.dropna()

def rhob(phi_rhob, Rho_matrix= 2650.0, Rho_fluid=1000.0):
    """
    Rho_matrix (sandstone) : 2.65 g/cc
    Rho_matrix (Limestome): 2.71 g/cc
    Rho_matrix (Dolomite): 2.876 g/cc
    Rho_matrix (Anyhydrite): 2.977 g/cc
    Rho_matrix (Salt): 2.032 g/cc

    Rho_fluid (fresh water): 1.0 g/cc (is this more mud-like?)
    Rho_fluid (salt water): 1.1 g/cc
    see wiki.aapg.org/Density-neutron_log_porosity
    returns density porosity log """
    
    return Rho_matrix*(1 - phi_rhob) + Rho_fluid*phi_rhob

phi_rhob = 2*(df.PHIND/100)/(1 - df.DeltaPHI/100) - df.DeltaPHI/100
calc_RHOB = rhob(phi_rhob)
df['RHOB'] = calc_RHOB

df.describe()

facies_dict = {1:'sandstone', 2:'c_siltstone', 3:'f_siltstone', 4:'marine_silt_shale',
               5:'mudstone', 6:'wackentstone', 7:'dolomite', 8:'packstone', 9:'bafflestone'}

df["s_Facies"] = df.Facies.map(lambda x: facies_dict[x])

df.head()

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

g = sns.PairGrid(df, hue="s_Facies", vars=['GR','RHOB','PE','ILD_log10'], size=4)

g.map_upper(plt.scatter,**dict(alpha=0.4))  
g.map_lower(plt.scatter,**dict(alpha=0.4))
g.map_diag(plt.hist,**dict(bins=20))  
g.add_legend()
g.set(alpha=0.5)

selected = ['f_siltstone', 'bafflestone', 'wackentstone']

dfs = pd.concat(list(map(lambda x: df[df.s_Facies == x], selected)))

g = sns.PairGrid(dfs, hue="s_Facies", vars=['GR','RHOB','PE','ILD_log10'], size=4)  
g.map_upper(plt.scatter,**dict(alpha=0.4))  
g.map_lower(plt.scatter,**dict(alpha=0.4))
g.map_diag(plt.hist,**dict(bins=20))  
g.add_legend()
g.set(alpha=0.5)

# Make X and y
X = dfs[['GR','ILD_log10','PE']].as_matrix()
y = dfs['Facies'].values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.3)

from sklearn.cluster import KMeans

clf = KMeans(n_clusters=4, random_state=1).fit(X)
y_pred = clf.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.3)

clf.inertia_

