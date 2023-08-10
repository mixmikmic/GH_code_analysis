#The main library we use. Manipulates Excel-like tables (called dataframes) with named rows and columns
import pandas as pd 
import numpy as np

#ploting capacities
import matplotlib.pyplot as plt 
#make plots appear in this notebook
get_ipython().magic('matplotlib inline')

#Default options for plots: this controls the font used in figures
font = {'family' : 'sans serif',
    'size'   : 18}
plt.rc('font', **font)

from res_ind_lib import *     #MAIN library: the functions used to compute risk, resilience, etc

df = pd.read_csv("results/all_data_and_results.csv", index_col=0, skiprows=[0,2])
df.head()

df.plot.scatter(x="gdp_pc_pp", y="pov_head", s=df["pop"]/5e3, alpha=0.5, figsize=(7,7))
plt.xlabel("Average income")
plt.ylabel("Poverty incidence")
plt.ylim(0);

plt.savefig("img/poverty_vs_income.png")

df.plot.scatter(x="pov_head", y="resilience", s=df["pop"]/5e3, alpha=0.5, figsize=(7,7))
plt.xlabel("Poverty incidence")
plt.ylabel("Socio-economic capacity")
plt.ylim(0);
plt.xlim(0);
plt.savefig("img/capacity_vs_poverty.png")

df.assign(vu=1/df.resilience).plot.scatter(x="pov_head", y="vu", s=df["pop"]/5e3, alpha=0.5, figsize=(7,7))
plt.xlabel("Poverty incidence")
plt.ylabel("Socio-economic vulnerability")
plt.ylim(0);
plt.xlim(0);
plt.savefig("img/se_vuln_vs_poverty.png")



