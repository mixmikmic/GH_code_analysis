get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

PDD = pd.read_table('PDD_6MV.csv', sep= ",", index_col=0).astype(np.float); # Load the PDD, convert strings to floats

PDD.index = PDD.index.map(int)
PDD.columns = PDD.columns.map(int)   # map values to floats 

PDD.index.rename('Depth_mm', inplace=True)  # rename the cols and index
PDD.columns.rename('Fields_mm', inplace=True)

PDD.head()

PDD.lookup([0], [10]).item()    # lookup depth and field

PDD[[20,100]].plot()

fields = PDD.columns.values.astype(np.float)
depths = PDD.index.values.astype(np.float)

PDDInterp = interpolate.interp2d(fields, depths, PDD.as_matrix(), kind='linear')  # create the 2D interp

def PDD_(field, depth):  # helper function to get a pdd
    return PDDInterp(field, depth).item()

PDD_(25.0, 3.5)

NPSF = pd.read_table('NPSF.csv', delimiter= ",").transpose()
NPSF.columns = NPSF.iloc[0]  #get columns columns
NPSF = NPSF.iloc[1:]
NPSF.head()

NPSF_interp = interpolate.interp1d(NPSF.index.values.astype(np.float), NPSF['NPSF'].values.astype(np.float)) 

def NPSF_(field):  # helper funtion to get the NPSF given a field size
    return NPSF_interp(field).item()

NPSF_(44.23)

def TPR(field_mm, depth_mm, depth_ref_mm):   # copied straigth from EXCEL
    # 'Calculate TPR for SSD f = 100cm'
    f = 1000.0   # The SSD under ref conditions, 100 cm
    S = field_mm
    d = depth_mm
    dr = depth_ref_mm
    correction = ((f + d) / (f + dr)) ** 2
    PDD_d = PDD_(((S * f) / (f + d)), d)
    PDD_dr = PDD_(((S * f) / (f + dr)), dr)
    NPSF_d = NPSF_(((S * f) / (f + d)))
    NPSF_dr = NPSF_(((S * f) / (f + dr)))
    TPR = (PDD_d / PDD_dr) * (NPSF_d / NPSF_dr) * correction
    return TPR

depth_ref_mm = 50.0 # the reference depth, cm
depth_mm = 100.0 # the depth to calculate TPR, cm
field_mm = 100.0  # calc for single field size, at surface under ref conditions
print(TPR(field_mm, depth_mm, depth_ref_mm))   # 0.84 agrees with EXCEL calcs

i = len(fields)
j = len(depths)
wilcox_data = np.zeros((i,j))  # init an empty array

j = 0
for field in fields:
    i = 0
    for depth in depths:
        A =  d[(d['structure'] == structure) & (d['metric'] == metric)]
        D =  A['diff']
        wilcox_data[j][i] = my_wilcox(D.values)
        i = i + 1
    j = j+ 1  

TPR_df = pd.DataFrame(data=wilcox_data, index=depths, columns=fields)  # 1st row as the column names

