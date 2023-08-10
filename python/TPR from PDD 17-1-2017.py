get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

import statsmodels.api as sm
from statsmodels import regression

PDD = pd.read_table('data/RC_truebeam_6X_raw_PDD.csv', sep= ",", index_col=0)#.astype(np.float); # Load the PDD, convert strings to floats

PDD.index = PDD.index.map(float)       # index and columns are ints for convenience later
PDD.columns = PDD.columns.map(float)   # map values to floats 

PDD.index.rename('Depth_cm', inplace=True)  # rename the cols and index
PDD.columns.rename('Fields_cm', inplace=True)

PDD.head()

PDD.shape

PDD.columns

PDD[[3.0, 10.0, 40.0]].plot()
plt.title('Red-A golden beam data')
plt.xlabel('Depth (cm)')
plt.ylabel('%DD')
plt.xlim(0,30)
plt.ylim(0,105)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fields = PDD.columns.values.astype(np.float)
depths = PDD.index.values.astype(np.float)

PDDInterp = interpolate.interp2d(fields, depths, PDD.as_matrix(), kind='linear')  # create the 2D interp

def PDD_(field, depth):  # helper function to get a pdd
    return PDDInterp(field, depth).item()

PDD_(field = 15, depth = 5)

# Get the NPSF data
NPSF_np = np.genfromtxt('data/NPSF.csv', delimiter= ",")
Fields_NPSF = NPSF_np[0][1:]
Fields_NPSF = Fields_NPSF/10.0   # convert to cm
NPSF = NPSF_np[1][1:]

#plt.plot(Fields_NPSF,NPSF)
Fields_NPSF

# Linear regression no good
x = Fields_NPSF
y = NPSF
X = sm.add_constant(x)

z = np.polyfit(x, y, deg = 3)   # do the poly fit, z is a list of the fit coefficients
p = np.poly1d(z)                # get a polyfit object, pass this an x to get the y

plt.scatter(x,y, label='raw NPSF')

xnew = np.arange(0, 50)
plt.plot(xnew,p(xnew), label='Poly fit to raw')
plt.xlabel('Field size')
plt.ylabel('NPSF')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

p(20)

NPSF_ = p    # just reassign as used later

NPSF_(20)

def TPR(field, depth, depth_ref):   # copied straigth from EXCEL, distances are cm
    # 'Calculate TPR for SSD f = 100cm'
    f = 100.0   # The SSD under ref conditions, 100 cm
    S = field
    d = depth
    dr = depth_ref
    correction = ((f + d) / (f + dr)) ** 2
    PDD_d = PDD_(((S * f) / (f + d)), d)
    PDD_dr = PDD_(((S * f) / (f + dr)), dr)
    NPSF_d = NPSF_(((S * f) / (f + d)))
    NPSF_dr = NPSF_(((S * f) / (f + dr)))
    TPR = (PDD_d / PDD_dr) * (NPSF_d / NPSF_dr) * correction
    return round(TPR, 3)

print(TPR(3, 5, 5))   # 0.84 agrees with EXCEL calcs

fields = PDD.columns.values.astype(np.float)
depths = PDD.index.values.astype(np.float)

columns = len(fields)
rows = len(depths)
TPR_array = np.zeros((rows,columns))  # init an empty array

PDD.columns.values#.astype(np.float)

TPR_array.shape

row = 0
for depth in depths:
    column = 0
    for field in fields:
        TPR_array[row][column] = TPR(field, depth, 5.0)
        column = column + 1
    row = row + 1  

TPR_df = pd.DataFrame(data=TPR_array, index=depths, columns=fields)  # 1st row as the column names

TPR_df.index.name = 'Depth_mm'
TPR_df.columns.name = 'Field_size'

TPR_df[1:2]

TPR_df.to_csv('Pandas_TPR.csv')

TPR_df.shape

Published_TPR = pd.read_table('data/Published_TPR_MJ.csv', sep= ",", index_col=0).astype(np.float); 

Published_TPR.index = Published_TPR.index.map(float)       # index and columns are ints for convenience later
Published_TPR.columns = Published_TPR.columns.map(float)   # map values to floats 

Published_TPR.index.rename('Depth_cm', inplace=True)  # rename the cols and index
Published_TPR.columns.rename('Fields_cm', inplace=True)

Published_TPR.head()

Published_TPR.shape

common_fields = list(set(Published_TPR.columns.values) & set(TPR_df.columns.values))
# common_fields

common_depths = list(set(Published_TPR.index.values) & set(TPR_df.index.values))
# common_depths

difference_df = TPR_df.loc[common_depths, common_fields] - Published_TPR.loc[common_depths, common_fields]
difference_df

difference_df.shape

difference_df.abs().max().max()

difference_df.abs().mean().mean()

difference_df_pct = 100.0*difference_df/Published_TPR.loc[common_depths, common_fields]
difference_df_pct.abs().max().max()

difference_df_pct.abs().mean().mean()

#plt.figure(figsize=(20, 5))
plt.imshow(difference_df_pct.abs().as_matrix())
plt.ylabel('Depth (mm)')
plt.colorbar()



