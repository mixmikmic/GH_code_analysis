import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import (Exchangeable,
    Independence,Autoregressive)
from statsmodels.genmod.families import Poisson

#!conda install statsmodels

ds = xr.open_dataset('/data/jennie/Pobs_start_season_1950_2016.nc')

starts = ds.starts

starts1 = ds.sel(X=-50,Y=14)

starts1a=starts1.starts.values

dMDR=xr.open_dataset('/data/jennie/Pkaplan_MDR_season_1950_2016.nc')

MDR=dMDR.MDR.values

dNINO3p4=xr.open_dataset('/data/jennie/Pkaplan_NINO3p4_season_1950_2016.nc')

NINO3p4=dNINO3p4.NINO3p4.values

dQBO=xr.open_dataset('/data/jennie/Pncep_QBO30mb_season_1950_2016.nc')

QBO=dQBO.QBO.values

subject= np.ones(67)

data = {'starts': starts1a,
        'MDR': MDR,
        'NINO3p4': NINO3p4,
         'QBO': QBO,'subject': subject}

y = list(range(0,67))

df = pd.DataFrame(data, index=[y])
df

fam = Poisson()
ind = Independence()
model1 = GEE.from_formula("starts ~ MDR + NINO3p4 + QBO", "subject", df, cov_struct=ind, family=fam)
result1 = model1.fit()
print(result1.summary())

result1.scale

result1.params

result1.params.MDR

result1



