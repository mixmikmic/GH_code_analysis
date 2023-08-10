import pandas as pd
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

import json

OVERNIGHT_FILE = '/home/buck06191/Desktop/optimisation_edit.json'
with open(OVERNIGHT_FILE) as f:
    optim_data = json.load(f)
# Check length of each dict section before converting to pandas DF
import copy
x = copy.copy(optim_data)
{k:len(x[k]) for k in x.keys()}

overnight_df = pd.DataFrame(optim_data)

def optimal_fit(xx, cortex):
    df = xx.loc[xx['Cortex']==cortex]
    return df.loc[df['Final_Distance']==df['Final_Distance'].min()]

df_PFC = overnight_df.loc[overnight_df['Cortex']=='PFC']

df_VC = overnight_df.loc[overnight_df['Cortex']=='VC']
optimal_PFC = df_PFC.loc[df_PFC.groupby(['Subject', 'Max_Demand']).Final_Distance.agg('idxmin')]
optimal_PFC

optimal_VC = df_VC.loc[df_VC.groupby(['Subject', 'Max_Demand']).Final_Distance.agg('idxmin')]


df = result = pd.concat([optimal_VC, optimal_PFC])
R_corr = df.groupby(['Cortex', 'Max_Demand'])['R_autc'].apply(lambda x: x.corr(df['R_autp']))
t_corr = df.groupby(['Cortex', 'Max_Demand'])['t_c'].apply(lambda x: x.corr(df['t_p']))
print(R_corr)
plt.figure()
plt.plot(R_corr.index.levels[1], R_corr.ix['PFC'], '.r', label='PFC')
plt.plot(R_corr.index.levels[1], R_corr.ix['VC'], '.b', label='VC')
plt.title('R Correlation')
plt.legend()

plt.figure()
plt.plot(t_corr.index.levels[1], t_corr.ix['PFC'], '.r', label='PFC')
plt.title('Time Correlation')
plt.plot(t_corr.index.levels[1], t_corr.ix['VC'], '.b', label='VC')
plt.legend()

g = sns.FacetGrid(df, col="Cortex", row='Max_Demand', hue='Subject')
g = (g.map(plt.scatter, "R_autp", "R_autc", edgecolor="w")).add_legend()

g = sns.FacetGrid(df, col="Cortex", row='Max_Demand', hue='Subject')
g = (g.map(plt.scatter, "t_p", "t_c", edgecolor="w")).add_legend()

g=sns.factorplot(data=overnight_df, x='Max_Demand', y='Final_Distance',
                 hue='Cortex', col='Cortex', kind='box', col_wrap=3)

param_list = ['R_autc', 't_c', 'R_autp', 't_p', 'R_autu', 't_u', 'R_auto', 't_o']
for parameter in param_list:
    plt.figure()
    g = sns.jointplot(parameter, 'Final_Distance', overnight_df)
    g.fig.suptitle(parameter)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
threedee = plt.figure().gca(projection='3d')
threedee.scatter(overnight_df['R_autp'], overnight_df['R_autc'], overnight_df['Final_Distance'])
threedee.set_xlabel('R_autp')
threedee.set_ylabel('R_autc')
threedee.set_zlabel('Final Distance')
plt.show()



