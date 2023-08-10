get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from astropy.table import Table

plt.style.use("fivethirtyeight")

dfstar = pd.read_csv("../data/star_identifier.csv")
dfobs = pd.read_csv("../data/observed.csv")
dfrv = pd.read_csv("../data/star_rv.csv")

objs = dfobs.groupby("objtype").get_group("obj")
print(len(objs))

dfobs.loc[dfobs.objtype=='obj', 'row_id'] =     [int(s.split("-")[1]) for s in objs.OBJECT.loc[dfobs.objtype=='obj']]

obsrv = pd.merge(
    dfobs,
    pd.merge(dfrv, dfstar, left_on='name', right_on='name', how='left'),
    how='left')

# quality counts for obeserved ta
obsrv.quality.value_counts()

