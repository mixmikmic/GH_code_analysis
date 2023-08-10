get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

PDD = pd.read_table('PDD_6MV.csv', sep= ",", index_col=0)#.astype(np.float); # Load the PDD, convert strings to floats

PDD.index = PDD.index.map(int)       # index and columns are ints for convenience later
PDD.columns = PDD.columns.map(int)   # map values to floats 

PDD.index.rename('Depth_mm', inplace=True)  # rename the cols and index
PDD.columns.rename('Fields_mm', inplace=True)

PDD[:15]

#PDD.tail()

PDD.shape

PDD[[20,100]].plot()

fields = PDD.columns.values.astype(np.float)
depths = PDD.index.values.astype(np.float)

PDDInterp = interpolate.interp2d(fields, depths, PDD.as_matrix(), kind='linear')  # create the 2D interp

def PDD_(field, depth):  # helper function to get a pdd
    return PDDInterp(field, depth).item()

PDD_(field = 25, depth = 3.5)

# Get the NPSF data
NPSF_np = np.genfromtxt('NPSF.csv', delimiter= ",")
Fields_NPSF = NPSF_np[0][1:]
NPSF = NPSF_np[1][1:]

#plt.plot(Fields_NPSF,NPSF)

# Linear regression no good
x = Fields_NPSF
y = NPSF
X = sm.add_constant(x)

model = regression.linear_model.OLS(y, X).fit()
intercept = model.params[0]
slope = model.params[1]

y2 = intercept + slope*x
plt.scatter(x,y)
plt.plot(x,y2)

z = np.polyfit(x, y, deg = 2)   # do the poly fit, z is a list of the fit coefficients
p = np.poly1d(z)                # get a polyfit object, pass this an x to get the y

plt.scatter(x,y, label='raw NPSF')

xnew = np.arange(-100, 500)
plt.plot(xnew,p(xnew), label='Poly fit to raw')
plt.xlabel('Field size')
plt.ylabel('NPSF')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

p(44.23)

NPSF_ = p    # just reassign as used later

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

print(TPR(field_mm = 5, depth_mm = 0 , depth_ref_mm = 350.0))   # working at extremes OK

fields = PDD.columns.values.astype(np.float)
depths = PDD.index.values.astype(np.float)

columns = len(fields)
rows = len(depths)
TPR_array = np.zeros((rows,columns))  # init an empty array

TPR_array.shape

row = 0
for depth in depths:
    column = 0
    for field in fields:
        TPR_array[row][column] = TPR(field, depth, depth_ref_mm)
        column = column + 1
    row = row + 1  

TPR_df = pd.DataFrame(data=TPR_array, index=depths, columns=fields)  # 1st row as the column names

TPR_df.index.name = 'Depth_mm'
TPR_df.columns.name = 'Field_size'

TPR_df.head()

TPR_df.to_csv('Pandas_TPR.csv')



